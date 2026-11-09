"""Single source of truth for v2 train and inference feature construction."""
from __future__ import annotations

import asyncio

from api import api_signals
from embedding_pipeline import embedding_vectors, project_document
from context_features import context_features
from scoring import (cosine, embedding_details, scores_for_email, v2_features)
from segments import KINDS, document_text, split_email


def collect_rows(rows, tfidf, centroids, cache, with_llm=False, verbose=False, zone_overrides=None, use_owned_context=False):
    """Rows are (subject, original_body, label). Work is batched and cached."""
    segments = ([zone_overrides[i] if zone_overrides[i] is not None else split_email(subject, body)
                 for i, (subject, body, _) in enumerate(rows)] if zone_overrides is not None else
                [split_email(subject, body) for subject, body, _ in rows])
    present = [(i, kind, zones[kind]) for i, zones in enumerate(segments)
               for kind in KINDS if zones[kind].strip()]
    tf_by_row = [{} for _ in rows]
    if tfidf is not None and present:
        texts = [text for _, _, text in present]
        tf_prob = tfidf['clf'].predict_proba(tfidf['vec'].transform(texts))[:, 1]
        for (i, kind, _), p in zip(present, tf_prob):
            tf_by_row[i][kind] = float(p)

    want_embed = centroids['backend'] == 'api'
    if with_llm:
        _, semantics = asyncio.run(api_signals(
            segments, want_embedding=False, want_semantic=True,
            cache=cache, verbose=verbose))
    else:
        semantics = {}

    embeddings = {}
    doc_vecs = [None] * len(rows)
    if want_embed:
        zone_keys = list(dict.fromkeys((kind, zones[kind]) for zones in segments
                                      for kind in KINDS if zones[kind].strip()))
        doc_texts = [document_text(zones) for zones in segments]
        inputs = [text for kind, text in zone_keys] + doc_texts
        vectors = embedding_vectors(inputs, cache=cache, verbose=verbose)
        embeddings = dict(zip(zone_keys, vectors[:len(zone_keys)]))
        doc_vecs = vectors[len(zone_keys):]

    centers = centroids['centroids']
    result = []
    for i, ((_, original_body, label), zones) in enumerate(zip(rows, segments)):
        emb_probs, sem_probs, similarities = {}, {}, {}
        for kind in KINDS:
            if not zones[kind].strip():
                continue
            vec = embeddings.get((kind, zones[kind]))
            similarities[kind] = embedding_details(
                vec, centers.get(f'{kind}_1'), centers.get(f'{kind}_0'))
            emb_probs[kind] = similarities[kind]['embedding']
            sem_probs[kind] = semantics.get((kind, zones[kind]))
        scores = scores_for_email(zones, tf_by_row[i], emb_probs, sem_probs, similarities)
        doc_detail = embedding_details(doc_vecs[i], centers.get('document_1'),
                                       centers.get('document_0'))
        doc_signals = {'document_embedding': doc_detail['embedding'],
                       'document_ai_similarity': doc_detail['embedding_ai_similarity'],
                       'document_human_similarity': doc_detail['embedding_human_similarity']}
        for n, v in enumerate(project_document(doc_vecs[i], centroids.get('projection'))):
            doc_signals[f'document_pca_{n}'] = v
        for name, a, b in (
            ('subject_body_coherence', 'subject', 'body'),
            ('greeting_body_coherence', 'greeting', 'body'),
            ('body_signature_coherence', 'body', 'signature'),
        ):
            sim = cosine(embeddings.get((a, zones[a])), embeddings.get((b, zones[b])))
            doc_signals[name] = sim
        context_body = ("\n".join(zones[k] for k in ('greeting', 'body', 'signature') if zones[k])
                        if use_owned_context else original_body)
        context = context_features(zones, context_body)
        result.append({'label': label, 'segments': zones, 'scores': scores,
                       'doc_signals': doc_signals, 'context': context,
                       'features_no_llm': v2_features(scores, context, doc_signals, include_llm=False),
                       'features_with_llm': v2_features(scores, context, doc_signals, include_llm=True)})
    return result

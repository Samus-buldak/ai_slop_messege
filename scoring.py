"""Единая математика для train и analyze: никаких разных формул."""
import math

from segments import KINDS


def mean_present(values):
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def embedding_score(vec, ai_center, human_center):
    """Псевдо-скор из нормализованных косинусов; НЕ калиброванная вероятность."""
    if vec is None or ai_center is None or human_center is None:
        return None
    if len(vec) != len(ai_center) or len(vec) != len(human_center):
        return None
    similarity_ai = sum(a * b for a, b in zip(vec, ai_center))
    similarity_human = sum(a * b for a, b in zip(vec, human_center))
    d = max(-50., min(50., 8. * (similarity_ai - similarity_human)))
    return 1 / (1 + math.exp(-d))


def scores_for_email(segs, tf_scores, emb_scores, sem_scores, emb_details=None):
    result = {}
    emb_details = emb_details or {}
    for kind in KINDS:
        present = bool(segs[kind].strip())
        tf = tf_scores.get(kind) if present else None
        em = emb_scores.get(kind) if present else None
        sm = sem_scores.get(kind) if present else None
        result[kind] = {'present': present, 'tfidf': tf, 'embedding': em,
                        'semantic_llm': sm, 'no_llm': mean_present((tf, em)),
                        'with_llm': mean_present((tf, em, sm)),
                        'embedding_ai_similarity': emb_details.get(kind, {}).get('embedding_ai_similarity'),
                        'embedding_human_similarity': emb_details.get(kind, {}).get('embedding_human_similarity')}
    return result


def features(scores):
    no_llm = [scores[k]['no_llm'] for k in KINDS]
    tfidf = [scores[k]['tfidf'] for k in KINDS]
    flags = [float(scores[k]['present']) for k in KINDS]
    return ([x if x is not None else 0. for x in no_llm] + flags +
            [mean_present(no_llm) or 0., mean_present(tfidf) or 0.])


def mean_zones(scores, mode):
    return mean_present([scores[k][mode] for k in KINDS])


def full_semantic_coverage(scores):
    return all(not scores[k]['present'] or scores[k]['semantic_llm'] is not None for k in KINDS)


def final_score(scores, tree_prob, mode):
    avg = mean_zones(scores, mode)
    if avg is None:
        return None
    return avg if tree_prob is None else (avg + float(tree_prob)) / 2


def feature_names():
    return ([f'score_{k}' for k in KINDS] + [f'has_{k}' for k in KINDS] +
            ['mean_score', 'mean_tf'])


# V2: Full-document Qwen embeddings, zone-wise similarities, LLM score and context
# go directly into the decision tree. The old helpers above remain for loading v1.
from context_features import CONTEXT_NAMES  # noqa: E402


V2_EMBED_FIELDS = ('embedding', 'embedding_ai_similarity', 'embedding_human_similarity')
V2_DOC_FIELDS = ('document_embedding', 'document_ai_similarity',
                 'document_human_similarity', 'subject_body_coherence',
                 'greeting_body_coherence', 'body_signature_coherence') + tuple(
                     f'document_pca_{i}' for i in range(8))


def cosine(a, b):
    if a is None or b is None or len(a) != len(b):
        return None
    return max(-1., min(1., sum(float(x) * float(y) for x, y in zip(a, b))))


def embedding_details(vector, ai_center, human_center):
    return {'embedding': embedding_score(vector, ai_center, human_center),
            'embedding_ai_similarity': cosine(vector, ai_center),
            'embedding_human_similarity': cosine(vector, human_center)}


def v2_feature_names():
    names = []
    for kind in KINDS:
        names += [f'{kind}_{x}' for x in ('present', 'tfidf', 'embedding',
                                         'embedding_ai_similarity', 'embedding_human_similarity',
                                         'semantic_llm', 'semantic_available')]
    names += list(V2_DOC_FIELDS)
    names += [f'{field}_available' for field in V2_DOC_FIELDS]
    names += list(CONTEXT_NAMES)
    return names


def v2_features(scores, context, doc_signals=None, include_llm=False):
    """Fixed feature layout, independent from presence of TF-IDF/semantic services."""
    values = []
    for kind in KINDS:
        s = scores[kind]
        sem = s.get('semantic_llm') if include_llm else None
        values.extend((float(s['present']),
                       s.get('tfidf') or 0., s.get('embedding') or 0.,
                       s.get('embedding_ai_similarity') or 0.,
                       s.get('embedding_human_similarity') or 0.,
                       sem if sem is not None else 0.,
                       float(sem is not None)))
    doc_signals = doc_signals or {}
    for field in V2_DOC_FIELDS:
        values.append(doc_signals.get(field) if doc_signals.get(field) is not None else 0.)
    for field in V2_DOC_FIELDS:
        values.append(float(doc_signals.get(field) is not None))
    values.extend(context.get(field, 0.) for field in CONTEXT_NAMES)
    assert len(values) == len(v2_feature_names())
    return [float(x) for x in values]


def final_score_v2(scores, tree_probability, mode):
    """Tree is the main decision maker; zone average is an auxiliary signal."""
    zone_average = mean_zones(scores, mode)
    if zone_average is None:
        return float(tree_probability)
    return .8 * float(tree_probability) + .2 * float(zone_average)

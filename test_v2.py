"""No LM Studio required: test input wiring with deterministic mocked API vectors."""
import json
from email.message import EmailMessage

import numpy as np
import pytest

import signals_v2
from context_features import CONTEXT_NAMES, context_features
from scoring import v2_feature_names, v2_features, scores_for_email
from segments import KINDS, document_text, parse_eml, split_email


def test_quoted_reply_and_signature_not_misattributed():
    text = ('Hello Michael,\n\nI reviewed the proposal today.\n\n'
            'Best regards,\nAlice\n\nOn Fri, Nov 6, 2026 at 10:23 AM Bob wrote:\n'
            '> This entire old email was written by somebody else.\n')
    s = split_email('Re: Plans', text)
    assert s['greeting'] == 'Hello Michael,'
    assert s['body'] == 'I reviewed the proposal today.'
    assert s['signature'] == 'Best regards,\nAlice'
    assert 'somebody else' not in document_text(s)


def test_html_quoted_thread_and_hidden_data_ignored():
    msg = EmailMessage()
    msg.add_alternative('''<p>Hello!</p><div>New message from me.</div>
        <div class="gmail_quote">On Tuesday Bob wrote: <p>HIDDEN HISTORY</p></div>
        <script>HIDDEN SCRIPT</script>''', subtype='html')
    subject, body = parse_eml(msg.as_bytes())
    assert 'New message from me' in body
    assert 'HIDDEN HISTORY' not in body and 'HIDDEN SCRIPT' not in body
    assert split_email(subject, body)['body'] == 'New message from me.'


def test_subject_and_body_context_features_stable():
    s = split_email('Report', 'Hi there!\n\nA short report.\n\nCheers,\nMary')
    values = context_features(s, 'Hi there!\n\nA short report.\n\nCheers,\nMary')
    assert tuple(values) == CONTEXT_NAMES
    assert all(0 <= x <= 1 for x in values.values())
    assert values['signature_present'] == 1


def test_llm_is_real_tree_input_not_just_postprocessing():
    zones = {'subject': 'test', 'greeting': '', 'body': 'body', 'signature': ''}
    score = scores_for_email(zones, {}, {'subject': .3, 'body': .4},
                             {'subject': .91, 'body': .92})
    context = context_features(zones, zones['body'])
    base = v2_features(score, context, include_llm=False)
    with_llm = v2_features(score, context, include_llm=True)
    idx = v2_feature_names().index('body_semantic_llm')
    assert len(base) == len(with_llm) == len(v2_feature_names())
    assert base[idx] == 0 and with_llm[idx] == pytest.approx(.92)
    assert base[v2_feature_names().index('body_semantic_available')] == 0
    assert with_llm[v2_feature_names().index('body_semantic_available')] == 1


def test_qwen_embeddings_document_coherence_are_features(monkeypatch):
    async def fake_api(segs, want_embedding=True, want_semantic=True, cache=None, verbose=False):
        emb, sem = {}, {}
        for s in segs:
            for k, text in s.items():
                if text:
                    if want_embedding:
                        emb[(k, text)] = [1., 0.] if 'assistant' in text.lower() else [0., 1.]
                    if want_semantic:
                        sem[(k, text)] = .9 if 'assistant' in text.lower() else .1
        return emb, sem

    def fake_embed(texts, **kwargs):
        return [[1., 0.] if 'assistant' in t.lower() else [0., 1.] for t in texts]

    monkeypatch.setattr(signals_v2, 'api_signals', fake_api)
    monkeypatch.setattr(signals_v2, 'embedding_vectors', fake_embed)
    centers = {'backend': 'api', 'centroids': {
        'document_1': [1., 0.], 'document_0': [0., 1.],
        'subject_1': [1., 0.], 'subject_0': [0., 1.],
        'body_1': [1., 0.], 'body_0': [0., 1.]}, 'model': 'mock-qwen'}
    rows = [('assistant update', 'assistant wrote this', 1),
            ('work update', 'just a quick human note', 0)]
    result = signals_v2.collect_rows(rows, None, centers, None, with_llm=True)
    assert result[0]['scores']['body']['embedding'] > result[1]['scores']['body']['embedding']
    assert result[0]['scores']['body']['semantic_llm'] == .9
    assert result[0]['doc_signals']['document_ai_similarity'] == pytest.approx(1.)
    assert result[0]['doc_signals']['subject_body_coherence'] == pytest.approx(1.)
    names = v2_feature_names()
    assert result[0]['features_with_llm'][names.index('body_semantic_llm')] == .9
    assert result[0]['features_no_llm'][names.index('body_semantic_llm')] == 0.
    assert result[0]['features_with_llm'][names.index('document_ai_similarity')] == 1.


def test_long_email_embedding_chunks_no_truncation(monkeypatch):
    import embedding_pipeline
    from api import _normalise_vector
    text = 'Long message with some details. ' * 300
    parts = embedding_pipeline.chunks(text)
    assert len(parts) > 1 and all(len(x) <= 3000 for x in parts)
    assert parts[-1].endswith('details.')

    async def fake_vectors(inputs, cache=None, verbose=False):
        return [[1., 0.] for _ in inputs]

    monkeypatch.setattr(embedding_pipeline, 'embedding_many', fake_vectors)
    assert embedding_pipeline.embedding_vectors([text, '']) == [[1., 0.], None]


def test_pca_embedding_projection_roundtrip():
    from embedding_pipeline import fit_document_projection, project_document
    rng = np.random.default_rng(18)
    vectors = rng.normal(size=(30, 12))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).tolist()
    pca = fit_document_projection(vectors)
    assert len(pca['components']) == 8
    assert len(project_document(vectors[0], pca)) == 8
    assert project_document(None, pca) == [None] * 8

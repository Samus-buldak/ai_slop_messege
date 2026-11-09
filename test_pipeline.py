import asyncio
import json
from email.message import EmailMessage

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import api
from scoring import embedding_score, features, full_semantic_coverage, mean_present, scores_for_email
from segments import KINDS, parse_eml, split_email
from server import app
from train import best_cut, split_rows


def test_mime_subject_body_html_and_attachment_ignored():
    msg = EmailMessage()
    msg['Subject'] = 'Задача — письмо'
    msg.set_content('Привет!\n\nОтправь отчёт.\n\nС уважением,\nКирилл')
    msg.add_attachment(b'SECRET BINARY ATTACHMENT', maintype='application',
                       subtype='octet-stream', filename='private.bin')
    subject, body = parse_eml(msg.as_bytes())
    zones = split_email(subject, body)
    assert subject == 'Задача — письмо'
    assert 'SECRET BINARY' not in body
    assert zones['greeting'] == 'Привет!'
    assert zones['body'] == 'Отправь отчёт.'
    assert zones['signature'].startswith('С уважением')
    assert tuple(zones) == KINDS


def test_html_only_and_no_leaked_scripts():
    msg = EmailMessage()
    msg['Subject'] = 'HTML only'
    msg.add_alternative('<style>.hidden{color:red}</style><script>HACK()</script>'
                        '<p>Здравствуйте!</p><div>Данные для письма</div>', subtype='html')
    _, body = parse_eml(msg.as_bytes())
    assert 'HACK' not in body and 'hidden' not in body
    assert 'Данные для письма' in body


def test_no_zero_padding_when_zones_absent():
    zones = {'subject': '', 'greeting': '', 'body': 'Только основное письмо', 'signature': ''}
    scores = scores_for_email(zones, {'body': 0.8}, {}, {})
    assert mean_present([scores[k]['no_llm'] for k in KINDS]) == 0.8
    assert features(scores)[8] == pytest.approx(0.8)
    assert not full_semantic_coverage(scores)
    assert embedding_score(None, [1., 0.], [0., 1.]) is None


def test_stratification_and_thresholds():
    rows = [(str(i), f'text {i}', label) for label in (0, 1) for i in range(120)]
    split = split_rows(rows, 17, 150, 120, 80)
    assert set(map(id, split['base'])).isdisjoint(map(id, split['test']))
    assert {x[2] for x in split['calib']} == {0, 1}
    cutoff, j = best_cut([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0.2 < cutoff < 0.8
    assert j == 1.0


def test_api_parse_strict_order():
    assert api._extract_scores('[{"i":1,"score":0.8},{"i":0,"score":0.2}]', 2) == [0.2, 0.8]
    with pytest.raises(ValueError):
        api._extract_scores('[{"i":0,"score":2.5}]', 1)
    with pytest.raises(ValueError):
        api._extract_scores('[{"i":0,"score":0.3},{"i":0,"score":0.4}]', 2)


def test_api_bad_batch_splits_without_deadlock(monkeypatch):
    monkeypatch.setattr(api, 'API_KEY', 'test-key')
    monkeypatch.setattr(api, 'API_URL', 'https://mock.invalid/v1')
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://mock.invalid/v1')
    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'test-key')
    calls = []

    def handler(request):
        assert request.headers['authorization'] == 'Bearer test-key'
        payload = json.loads(request.content)
        assert request.url.path.endswith('/chat/completions')
        data = json.loads(payload['messages'][1]['content'])
        calls.append(len(data))
        if len(data) > 1:
            return httpx.Response(200, json={'choices': [{'message': {'content': 'broken json'}}]})
        return httpx.Response(200, json={'choices': [
            {'message': {'content': json.dumps([{'i': 0, 'score': 0.7}])}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await asyncio.wait_for(api.semantic_many([('body', f'text {i}') for i in range(8)],
                                                           client=client), timeout=3)

    scores = asyncio.run(run())
    assert scores == [0.7] * 8
    assert calls[0] == 8 and 1 in calls


def test_embedding_api_actual_vectors_not_hash(monkeypatch):
    monkeypatch.setattr(api, 'API_KEY', 'test-key')
    monkeypatch.setattr(api, 'API_URL', 'https://mock.invalid/v1')

    def handler(request):
        data = json.loads(request.content)
        assert request.url.path.endswith('/embeddings')
        return httpx.Response(200, json={'data': [
            {'index': i, 'embedding': [3.0, 4.0]} for i, _ in enumerate(data['input'])]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await api.embedding_many(['one', 'two'], client=client)
    embeddings = asyncio.run(run())
    assert len(embeddings) == 2
    assert np.linalg.norm(embeddings[0]) == pytest.approx(1)
    assert embeddings[0] == pytest.approx([0.6, 0.8])


def test_http_raw_multipart_and_413():
    app_client = TestClient(app)
    raw = (b'Subject: hello\nContent-Type: text/plain; charset=utf-8\n\n'
           b'Hi!\n\nPlease read the attached document.')
    raw_response = app_client.post('/analyze', content=raw, headers={'Content-Type': 'message/rfc822'})
    file_response = app_client.post('/analyze', files={'file': ('sample.eml', raw)})
    assert raw_response.status_code == 200
    assert file_response.status_code == 200
    assert raw_response.json()['final_score'] == file_response.json()['final_score']
    assert app_client.post('/analyze', content=b'a' * (5 * 1024 * 1024 + 1)).status_code == 413

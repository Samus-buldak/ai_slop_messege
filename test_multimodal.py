"""No real API or LM Studio needed: all HTTP uses httpx mock transports."""
import asyncio
import io
import json
from email.message import EmailMessage

import httpx
import pytest
from PIL import Image

import analyze
import api
import segmentation_llm
import vision
from media_images import extract_images
from segments import split_email


def png_bytes(color='blue'):
    image = Image.new('RGB', (120, 80), color)
    buf = io.BytesIO()
    image.save(buf, 'PNG')
    return buf.getvalue()


def make_email(with_image):
    message = EmailMessage()
    message['Subject'] = 'Привет'
    message.set_content('Привет, коллега\n\nПроверь сообщение.\n\nУдачи,\nОлег')
    if with_image:
        message.add_attachment(png_bytes(), maintype='image', subtype='png', filename='photo.png')
    return message.as_bytes()


def test_images_from_email_only_and_metadata_sanitized():
    raw = make_email(True)
    images, warnings = extract_images(raw)
    assert len(images) == 1 and not warnings
    assert images[0].jpeg.startswith(b'\xff\xd8')
    assert (images[0].width, images[0].height) == (120, 80)
    assert len(extract_images(make_email(False))[0]) == 0


def test_vision_only_when_image_present_and_correct_routing(monkeypatch):
    monkeypatch.setattr(api, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(api, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(vision, 'VISION_MODEL', 'vision-capable-model')
    images, _ = extract_images(make_email(True))
    calls = []

    def handler(request):
        calls.append(request)
        payload = json.loads(request.content)
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer unit-test-token'
        assert payload['model'] == 'vision-capable-model'
        media = payload['messages'][1]['content'][1]
        assert media['type'] == 'image_url'
        assert media['image_url']['url'].startswith('data:image/jpeg;base64,')
        return httpx.Response(200, json={'choices': [{'message': {'content':
            json.dumps({'assessable': True, 'score': 0.8, 'reason': 'Visual anomalies'})}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await vision.assess_images([], client=client) == []
            outcome = await vision.assess_images(images, client=client)
        return outcome

    outcome = asyncio.run(run())
    assert len(calls) == 1
    assert outcome[0]['score'] == pytest.approx(.8)
    assert vision.combine_scores(.4, [.8]) == pytest.approx(.48)
    assert vision.combine_scores(.4, []) == pytest.approx(.4)


def test_vision_invalid_json_no_fake_score(monkeypatch):
    monkeypatch.setattr(api, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(api, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_KEY', 'unit-test-token')
    monkeypatch.setattr(vision, 'VISION_API_URL', 'https://vision.example/v1')
    monkeypatch.setattr(vision, 'VISION_MODEL', 'vision-capable-model')
    images, _ = extract_images(make_email(True))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'choices': [{'message': {'content': 'not json'}}]}))) as client:
            return await vision.assess_images(images, client=client)
    assert asyncio.run(run())[0]['score'] is None


def test_llm_segmentation_uses_indices_without_rewriting(monkeypatch):
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_MODEL', 'test-segmenter')
    text = 'Good day Pavel\n\nLong message with details.\n\nYour friend, Alex'
    events = []

    def handler(request):
        payload = json.loads(request.content)
        assert request.url.path == '/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer unit-test-token'
        data = json.loads(payload['messages'][1]['content'])
        events.append(data)
        answers = [{'i': d['i'], 'greeting_end': 2, 'signature_start': 4} for d in data]
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(answers)}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await segmentation_llm.segment_many([('Title', text)], client=client)

    result = asyncio.run(run())
    zones, llm_used = result[0]
    assert llm_used and len(events) == 1
    assert zones['greeting'] == 'Good day Pavel'
    assert zones['body'] == 'Long message with details.'
    assert zones['signature'] == 'Your friend, Alex'
    assert zones['subject'] == 'Title'
    assert 'Long message with details.' in text


def test_bad_segmentation_is_fallback(monkeypatch):
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://llm.example/v1')
    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_API_KEY', 'unit-test-token')
    monkeypatch.setattr(segmentation_llm, 'SEMANTIC_MODEL', 'test-segmenter')
    text = 'Hi Pavel\n\nPlease read.\n\nBest regards,\nA.'

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'choices': [{'message': {'content':
                '[{"i":0,"greeting_end":300,"signature_start":-1}]'}}]}))) as client:
            return await segmentation_llm.segment_many([('Subject', text)], client=client)
    zones, used = asyncio.run(run())[0]
    assert not used
    assert zones == split_email('Subject', text)


def test_combination_in_inference_no_image_does_not_call_vision(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, 'MODEL_DIR', tmp_path)
    monkeypatch.setattr(analyze, 'load_model', lambda: (None, {'no_llm': object()}, {'schema_version': 2}, {}))
    monkeypatch.setattr(analyze, '_score_email_v2', lambda raw, model: {
        'final_score': .6, 'verdict': 'AI', 'engine_thresholds': {
            'human_cut': .25, 'ai_cut': .55}, 'warnings': [], 'segments': {}})
    calls = []

    async def fake_vision(images, cache):
        calls.append(len(images))
        return [{'score': .9, 'reason': 'test'} for _ in images]

    monkeypatch.setattr(analyze, 'assess_images', fake_vision)
    no_image = analyze.score_email(make_email(False))
    assert calls == []
    assert no_image['final_score'] == .6
    with_image = analyze.score_email(make_email(True))
    assert calls == [1]
    assert with_image['photos']['count'] == 1
    assert with_image['photos']['checked'] == 1
    assert with_image['final_score'] == pytest.approx(.66)
    assert with_image['text_final_score'] == .6
    assert with_image['photos']['fusion_calibrated'] is False

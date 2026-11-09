"""Hybrid uses the local embeddings endpoint and external chat only when explicitly opted in."""
import asyncio
import json

import httpx
import pytest

import api
import local_qwen


def test_routed_embedding_and_semantic_have_distinct_auth_and_endpoints(monkeypatch):
    monkeypatch.setattr(api, 'API_URL', 'http://127.0.0.1:1234/v1')
    monkeypatch.setattr(api, 'API_KEY', 'lm-studio')
    monkeypatch.setattr(api, 'EMBED_MODEL', 'qwen3-embedding-local')
    monkeypatch.setattr(api, 'SEMANTIC_API_URL', 'https://remote.invalid/api/v1')
    monkeypatch.setattr(api, 'SEMANTIC_API_KEY', 'test-secret-not-a-real-key')
    monkeypatch.setattr(api, 'SEMANTIC_MODEL', 'chat-example')
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append((request.url.host, request.url.path, request.headers['authorization'], payload['model']))
        if request.url.path.endswith('/embeddings'):
            return httpx.Response(200, json={'data': [
                {'index': i, 'embedding': [3.0, 4.0]} for i in range(len(payload['input']))]})
        assert request.url.path.endswith('/chat/completions')
        data = json.loads(payload['messages'][1]['content'])
        return httpx.Response(200, json={'choices': [{'message': {'content':
            json.dumps([{'i': i, 'score': .6} for i in range(len(data))])}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            embeds = await api.embedding_many(['hello'], client=client)
            sem = await api.semantic_many([('body', 'hello')], client=client)
        return embeds, sem
    embeds, sem = asyncio.run(run())
    assert embeds[0] == pytest.approx([.6, .8])
    assert sem == [.6]
    assert ('127.0.0.1', '/v1/embeddings', 'Bearer lm-studio', 'qwen3-embedding-local') in seen
    assert ('remote.invalid', '/api/v1/chat/completions', 'Bearer test-secret-not-a-real-key', 'chat-example') in seen


def test_local_qwen_env_chat_separate_from_embeddings(monkeypatch, tmp_path):
    (tmp_path / '.env').write_text('API_URL=https://chat.invalid/api/v1\n'
                                   'API_KEY=secret-example\nMODEL=QwenChat\n', encoding='utf8')
    monkeypatch.setattr(local_qwen, 'ROOT', tmp_path)
    for key in ('API_URL', 'API_KEY', 'MODEL', 'SEMANTIC_API_URL',
                'SEMANTIC_API_KEY', 'SEMANTIC_MODEL'):
        monkeypatch.delenv(key, raising=False)
    local_qwen.configure(llm_from_env=True)
    import os
    assert os.environ['API_URL'] == 'http://127.0.0.1:1234/v1'
    assert os.environ['EMBED_MODEL'] == 'qwen3-embedding-local'
    assert os.environ['SEMANTIC_API_URL'] == 'https://chat.invalid/api/v1'
    assert os.environ['SEMANTIC_API_KEY'] == 'secret-example'
    assert os.environ['SEMANTIC_MODEL'] == 'QwenChat'


def test_local_qwen_without_env_llm_never_uses_remote(monkeypatch, tmp_path):
    (tmp_path / '.env').write_text('API_URL=https://chat.invalid/api/v1\n'
                                   'API_KEY=secret-example\nMODEL=QwenChat\n', encoding='utf8')
    monkeypatch.setattr(local_qwen, 'ROOT', tmp_path)
    for key in ('API_URL', 'API_KEY', 'MODEL', 'SEMANTIC_API_URL',
                'SEMANTIC_API_KEY', 'SEMANTIC_MODEL', 'LOCAL_LLM_MODEL'):
        monkeypatch.delenv(key, raising=False)
    local_qwen.configure()
    import os
    assert os.environ['SEMANTIC_API_URL'] == 'http://127.0.0.1:1234/v1'
    assert os.environ['SEMANTIC_API_KEY'] == 'lm-studio'
    assert os.environ['SEMANTIC_MODEL'] == 'unused-local-semantic-disabled'

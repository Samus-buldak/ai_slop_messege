"""Run local Qwen embeddings, optionally with LM Studio chat or chat API from .env."""
import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'qwen3-embedding-local'
MODEL_KEY = 'text-embedding-qwen3-embedding-0.6b'
LOCAL_ENDPOINT = 'http://127.0.0.1:1234/v1'
MODEL_FOLDER = 'model_hybrid_qwen'  # Does not overwrite previous model_local_qwen/


def _original_chat_settings():
    """Read only the original .env chat credentials before local override.

    Environment variables override .env, as in config.py. Never print secrets.
    """
    values = {}
    env_file = ROOT / '.env'
    if env_file.is_file():
        for raw in env_file.read_text(encoding='utf-8-sig').splitlines():
            line = raw.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                values[k.strip()] = v.strip().strip('\"\'')
    return {k: os.environ.get(k, values.get(k, '')) for k in ('API_URL', 'API_KEY', 'MODEL')}


def configure(chat_model=None, llm_from_env=False):
    remote = _original_chat_settings() if llm_from_env else None
    if remote:
        if not remote['API_KEY'] or remote['API_KEY'].lower() in {
                'sk-...', 'sk-xxx', 'your_api_key_here', 'changeme'}:
            raise RuntimeError('For --llm-from-env, set a real API_KEY in the original .env')
        if not remote['API_URL'] or not remote['MODEL']:
            raise RuntimeError('For --llm-from-env, set API_URL and MODEL in the original .env')
    local_chat = chat_model or os.environ.get('LOCAL_LLM_MODEL') or 'unused-local-semantic-disabled'
    os.environ.update(API_URL=LOCAL_ENDPOINT, API_KEY='lm-studio',
                      EMBED_MODEL=MODEL_ID, MODEL=local_chat,
                      SEMANTIC_API_URL=(remote['API_URL'].rstrip('/') if remote else LOCAL_ENDPOINT),
                      SEMANTIC_API_KEY=(remote['API_KEY'] if remote else 'lm-studio'),
                      SEMANTIC_MODEL=(remote['MODEL'] if remote else local_chat),
                      EMB_BATCH='8', EMB_CONC='1', SEM_BATCH='2', SEM_CONC='1',
                      API_TIMEOUT='120', API_RETRIES='1',
                      DETECTOR_MODEL_DIR=MODEL_FOLDER)


def _probe_embedding():
    import httpx
    try:
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.post(LOCAL_ENDPOINT + '/embeddings', json={
                'model': MODEL_ID, 'input': ['Local Qwen embedding connection test.']})
            response.raise_for_status()
            vector = response.json()['data'][0]['embedding']
    except (httpx.HTTPError, KeyError, TypeError, ValueError, IndexError):
        return None
    if len(vector) != 1024:
        raise RuntimeError(f'Expected Qwen 1024-dim embedding, received {len(vector)}')
    return vector


def ensure_embedding():
    if _probe_embedding() is not None:
        print('OK: local Qwen embeddings (1024 dimensions).', file=sys.stderr, flush=True)
        return
    lms = shutil.which('lms')
    if not lms:
        candidate = Path.home() / '.lmstudio/bin/lms.exe'
        lms = str(candidate) if candidate.is_file() else None
    if not lms:
        raise RuntimeError('Qwen endpoint did not respond, and LM Studio CLI was not found. '
                           'Start LM Studio local server and load qwen3-embedding-local.')
    # Server may be up, but the embedding model not loaded.
    with __import__('httpx').Client(timeout=10, trust_env=False) as client:
        try:
            response = client.get(LOCAL_ENDPOINT + '/models')
            response.raise_for_status()
        except __import__('httpx').HTTPError:
            subprocess.run([lms, 'server', 'start', '--port', '1234', '--bind', '127.0.0.1'], check=True)
    loaded = subprocess.run([lms, 'ps', '--json'], capture_output=True, text=True, check=True)
    running = json.loads(loaded.stdout)
    if isinstance(running, dict):
        running = running.get('models', running.get('loaded', []))
    if not any(m.get('identifier') == MODEL_ID for m in running):
        subprocess.run([lms, 'load', MODEL_KEY, '--identifier', MODEL_ID,
                        '--gpu', 'max', '--context-length', '4096', '-y'], check=True)
    if _probe_embedding() is None:
        raise RuntimeError('Local Qwen embeddings did not respond after loading. '
                           'Check LM Studio server and model identifier.')
    print('OK: local Qwen embeddings (1024 dimensions).', file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'train', 'text', 'analyze', 'server'])
    parser.add_argument('--llm-model', help='Local chat model identifier loaded in LM Studio, '
                                           'e.g. qwen3-4b-instruct; embedding model is separate')
    parser.add_argument('--llm-from-env', action='store_true',
                        help='Use API_URL, API_KEY, MODEL from original .env for chat ONLY; '
                             'the email segments will be sent to that external service')
    args, rest = parser.parse_known_args()
    os.chdir(ROOT)
    if args.llm_from_env and args.llm_model:
        parser.error('--llm-from-env and --llm-model cannot be used together')
    configure(chat_model=args.llm_model, llm_from_env=args.llm_from_env)
    ensure_embedding()
    if args.command == 'check':
        return
    if args.command == 'train':
        if '--offline' in rest or '--no-llm' in rest:
            parser.error('Local Qwen training needs embeddings; omit --offline/--no-llm.')
        if '--with-llm' in rest and not args.llm_from_env and os.environ['MODEL'] == 'unused-local-semantic-disabled':
            parser.error('Use --llm-from-env to activate your original .env chat model; '
                         'otherwise set LOCAL_LLM_MODEL or --llm-model <loaded-chat-model-id>.')
        sys.argv = ['train.py', *rest]
        runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
        saved = json.loads((ROOT / MODEL_FOLDER / 'centroids.json').read_text(encoding='utf-8'))
        if saved['backend'] != 'api' or saved['model'] != MODEL_ID:
            raise RuntimeError('Training did not produce Qwen embedding centroids.')
    else:
        path = ROOT / MODEL_FOLDER / 'centroids.json'
        if not path.exists():
            raise RuntimeError('Train first: local_qwen.py train')
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['backend'] != 'api' or saved['model'] != MODEL_ID:
            raise RuntimeError('Incompatible Qwen model artifacts; run local_qwen.py train again.')
        if args.command == 'server':
            import uvicorn
            uvicorn.run('server:app', host='127.0.0.1', port=8000)
        else:
            script = 'test_my_text.py' if args.command == 'text' else 'analyze.py'
            sys.argv = [script, *rest]
            runpy.run_path(str(ROOT / script), run_name='__main__')


if __name__ == '__main__':
    main()

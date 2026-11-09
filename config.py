"""Настройки. Переменные окружения имеют приоритет над локальным .env."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for raw_line in (ROOT / '.env').read_text(encoding='utf-8').splitlines() if (ROOT / '.env').is_file() else []:
    line = raw_line.strip()
    if line and not line.startswith('#') and '=' in line:
        key, value = line.split('=', 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"\'')

API_KEY = os.getenv('API_KEY', '').strip()
if API_KEY.lower() in {'', 'sk-...', 'sk-xxx', 'your_api_key_here', 'changeme'}:
    API_KEY = ''
API_URL = os.getenv('API_URL', 'https://deepcode.ci.nsu.ru/api/v1').rstrip('/')
MODEL = os.getenv('MODEL', 'Qwen3.8-Flash-Next')
EMBED_MODEL = os.getenv('EMBED_MODEL', 'text-embedding-3-small')
# Independent providers: local embeddings + optionally a remote chat model.
# Plain train.py keeps the original single-provider .env behavior by default.
SEMANTIC_API_URL = os.getenv('SEMANTIC_API_URL', API_URL).rstrip('/')
SEMANTIC_API_KEY = os.getenv('SEMANTIC_API_KEY', API_KEY).strip()
if SEMANTIC_API_KEY.lower() in {'', 'sk-...', 'sk-xxx', 'your_api_key_here', 'changeme'}:
    SEMANTIC_API_KEY = ''
SEMANTIC_MODEL = os.getenv('SEMANTIC_MODEL', MODEL).strip()
VISION_MODEL = os.getenv('VISION_MODEL', SEMANTIC_MODEL).strip()
VISION_API_URL = os.getenv('VISION_API_URL', SEMANTIC_API_URL).rstrip('/')
VISION_API_KEY = os.getenv('VISION_API_KEY', SEMANTIC_API_KEY).strip()
SEM_BATCH = max(1, int(os.getenv('SEM_BATCH', '8')))
SEM_CONC = max(1, min(32, int(os.getenv('SEM_CONC', '8'))))
EMB_BATCH = max(1, int(os.getenv('EMB_BATCH', '32')))
EMB_CONC = max(1, min(32, int(os.getenv('EMB_CONC', '6'))))
API_TIMEOUT = max(5., float(os.getenv('API_TIMEOUT', '35')))
API_RETRIES = max(0, min(5, int(os.getenv('API_RETRIES', '2'))))
MODEL_DIR = ROOT / os.getenv('DETECTOR_MODEL_DIR', 'model')
DATA_DIR = ROOT / 'data'

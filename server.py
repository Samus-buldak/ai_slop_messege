"""POST /analyze: raw RFC822 .eml или multipart file. GET /health."""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from starlette.datastructures import UploadFile
from starlette.concurrency import run_in_threadpool

from analyze import score_email
from config import API_KEY, MODEL_DIR

app = FastAPI(title='Kirusha AI Email Detector', version='2.0')
MAX_EMAIL_BYTES = 5 * 1024 * 1024


@app.get('/health')
def health():
    expected = ('tfidf.pkl', 'tree.pkl', 'thresholds.json', 'centroids.json')
    return {'status': 'ok', 'model_ready': all((MODEL_DIR / x).is_file() for x in expected),
            'api_key_set': bool(API_KEY)}


@app.post('/analyze')
async def analyze(request: Request):
    content_type = request.headers.get('content-type', '')
    if content_type.startswith('multipart/form-data'):
        async with request.form() as form:
            file = form.get('file')
            if not isinstance(file, UploadFile):
                raise HTTPException(status_code=400, detail='Нужен multipart file (поле file)')
            raw = await file.read(MAX_EMAIL_BYTES + 1)
    else:
        pieces = []
        count = 0
        async for piece in request.stream():
            count += len(piece)
            if count > MAX_EMAIL_BYTES:
                raise HTTPException(status_code=413, detail='Email больше 5 MB')
            pieces.append(piece)
        raw = b''.join(pieces)
    if len(raw) > MAX_EMAIL_BYTES:
        raise HTTPException(status_code=413, detail='Email больше 5 MB')
    if not raw.strip():
        raise HTTPException(status_code=400, detail='Пустое письмо')
    try:
        # Важно: не блокировать event loop HTTP-запросами и sklearn inference.
        return await run_in_threadpool(score_email, raw)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

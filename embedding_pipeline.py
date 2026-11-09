"""Real Qwen embedding vectors for short and long texts, cached per chunk."""
from __future__ import annotations

import asyncio
import math

from api import embedding_many

MAX_CHARS = 3000  # Less than API's hard 4000-character cap
OVERLAP = 180


def chunks(text: str):
    """Overlapping chunks, with sentence / space-aware boundaries when possible."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHARS:
        return [text]
    result = []
    start = 0
    while start < len(text):
        stop = min(len(text), start + MAX_CHARS)
        if stop < len(text):
            # Split at a natural whitespace boundary if we won't lose too much text.
            boundary = max(text.rfind('\n', start + MAX_CHARS // 2, stop),
                           text.rfind(' ', start + MAX_CHARS // 2, stop))
            if boundary > start:
                stop = boundary + 1
        piece = text[start:stop].strip()
        if piece:
            result.append(piece)
        if stop >= len(text):
            break
        start = max(start + 1, stop - OVERLAP)
    return result


def _unit_mean(vectors):
    if not vectors or any(v is None for v in vectors):
        return None
    if len({len(v) for v in vectors}) != 1:
        return None
    avg = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]
    norm = math.sqrt(sum(x * x for x in avg))
    return [x / norm for x in avg] if norm > 0 else None


def embedding_vectors(texts, cache=None, verbose=False):
    """Returns one unit vector per input. No fabricated fallback for failures."""
    indices, flat = [], []
    for text in texts:
        parts = chunks(text)
        indices.append((len(flat), len(flat) + len(parts)))
        flat.extend(parts)
    if not flat:
        return [None] * len(texts)
    vectors = asyncio.run(embedding_many(flat, cache=cache, verbose=verbose))
    return [_unit_mean(vectors[a:b]) for a, b in indices]


def fit_document_projection(vectors, max_components=8):
    """Fit supervised-model-independent PCA only on base-split document vectors."""
    import numpy as np
    from sklearn.decomposition import PCA

    valid = [v for v in vectors if v is not None]
    if len(valid) < 5 or len({len(v) for v in valid}) != 1:
        return None
    data = np.asarray(valid, dtype=np.float64)
    n = min(max_components, len(valid) - 1, data.shape[1])
    projection = PCA(n_components=n, random_state=42, svd_solver='auto')
    projection.fit(data)
    # Scores are standardized by training-set variance, then clipped on inference.
    return {'mean': projection.mean_.tolist(),
            'components': projection.components_.tolist(),
            'scale': np.maximum(np.sqrt(projection.explained_variance_), 1e-6).tolist()}


def project_document(vector, projection, max_components=8):
    """Return fixed-length PCA features, None for missing components."""
    import numpy as np

    if vector is None or not projection or len(vector) != len(projection['mean']):
        return [None] * max_components
    x = np.asarray(vector, dtype=np.float64) - np.asarray(projection['mean'])
    components = np.asarray(projection['components'])
    scale = np.asarray(projection['scale'])
    if components.ndim != 2 or components.shape[1] != len(x):
        return [None] * max_components
    transformed = (components @ x) / scale
    return [float(np.clip(v, -4, 4)) for v in transformed[:max_components]] + [None] * max(0, max_components - len(transformed))

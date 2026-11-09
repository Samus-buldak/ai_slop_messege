"""Extract actual MIME image attachments safely; never fetch remote image URLs.

Resize/transpose/encode as JPEG to drop EXIF, GPS and other metadata before
sending a vision model a data: URI. No image API is called by this module.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from email import policy
from email.message import Message
from email.parser import BytesParser

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGES = 4
MAX_ORIGINAL_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 12_000_000
MAX_SIDE = 1280


@dataclass(frozen=True)
class Photo:
    sha256: str
    jpeg: bytes
    width: int
    height: int
    filename: str


def prepare_photo(data: bytes, filename: str = '') -> Photo | None:
    if not data or len(data) > MAX_ORIGINAL_BYTES:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {'PNG', 'JPEG', 'WEBP', 'GIF', 'BMP', 'TIFF'}:
                return None
            w, h = image.size
            if w < 48 or h < 48 or w * h > MAX_PIXELS:
                return None  # reject tracking pixels and image decompression bombs
            image.seek(0)  # animation: first frame only
            frame = ImageOps.exif_transpose(image)
            frame.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            if frame.mode in ('RGBA', 'LA') or 'transparency' in frame.info:
                rgba = frame.convert('RGBA')
                rgb = Image.new('RGB', rgba.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.getchannel('A'))
            else:
                rgb = frame.convert('RGB')
            buffer = io.BytesIO()
            rgb.save(buffer, format='JPEG', quality=86, optimize=True)
            normalized = buffer.getvalue()
            return Photo(hashlib.sha256(normalized).hexdigest(), normalized,
                         rgb.width, rgb.height, filename[:120])
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning,
            OSError, ValueError, SyntaxError, EOFError):
        return None


def extract_images(raw: bytes | str) -> tuple[list[Photo], list[str]]:
    if isinstance(raw, str):
        raw = raw.encode('utf-8', 'replace')
    message: Message = BytesParser(policy=policy.default).parsebytes(raw)
    photos: list[Photo] = []
    warnings: list[str] = []
    unique = set()
    total = 0
    for part in message.walk():
        if part.is_multipart() or not part.get_content_type().startswith('image/'):
            continue
        total += 1
        if len(photos) >= MAX_IMAGES:
            continue
        try:
            payload = part.get_payload(decode=True) or b''
        except (TypeError, ValueError):
            continue
        photo = prepare_photo(payload, part.get_filename() or '')
        if photo is None:
            warnings.append('Одно из изображений пропущено: неподдерживаемый формат, размер или повреждение.')
        elif photo.sha256 not in unique:
            unique.add(photo.sha256)
            photos.append(photo)
    if total > MAX_IMAGES:
        warnings.append(f'Обработаны максимум {MAX_IMAGES} вложенных изображения; остальные пропущены.')
    return photos, warnings

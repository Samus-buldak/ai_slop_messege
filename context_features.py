"""Interpretable, bounded contextual features (not authorship proof)."""
from __future__ import annotations

import math
import re

from segments import KINDS

CONTEXT_NAMES = (
    'log_document_words', 'log_body_words', 'log_subject_words',
    'mean_token_length', 'lexical_diversity', 'average_sentence_words',
    'question_per_sentence', 'exclamation_per_sentence', 'comma_per_word',
    'digit_fraction', 'upper_fraction', 'newline_per_word',
    'bullet_line_fraction', 'url_count', 'quote_line_fraction',
    'has_reply_marker', 'has_forward_marker', 'greeting_present',
    'signature_present', 'subject_present', 'body_present',
)
TOKEN_RE = re.compile(r"(?u)\b[\w'-]+\b")
REPLY_RE = re.compile(r'(?im)^(?:on .+ wrote:|в .+ писал[аи]?:|>\s*|от:\s*|from:\s*)')
FORWARD_RE = re.compile(r'(?im)^[-\s]*(?:forwarded message|original message|пересланное сообщение)')


def _clamp(value):
    return max(0.0, min(1.0, float(value)))


def context_features(zones: dict[str, str], original_body: str = '') -> dict[str, float]:
    text = '\n'.join(zones[k] for k in KINDS if zones[k])
    tokens = TOKEN_RE.findall(text)
    body_tokens = TOKEN_RE.findall(zones['body'])
    subject_tokens = TOKEN_RE.findall(zones['subject'])
    words = max(1, len(tokens))
    sentences = max(1, len(re.findall(r'[.!?。！？]+(?:\s|$)', text)))
    letters = [c for c in text if c.isalpha()]
    body_lines = [line for line in original_body.splitlines() if line.strip()]
    result = (
        _clamp(math.log1p(len(tokens)) / 9),
        _clamp(math.log1p(len(body_tokens)) / 9),
        _clamp(math.log1p(len(subject_tokens)) / 6),
        _clamp(sum(map(len, tokens)) / words / 12),
        _clamp(len(set(t.casefold() for t in tokens)) / words),
        _clamp(words / sentences / 50),
        _clamp(text.count('?') / sentences / 3),
        _clamp(text.count('!') / sentences / 3),
        _clamp(text.count(',') / words / .2),
        _clamp(sum(c.isdigit() for c in text) / max(1, len(text)) / .3),
        _clamp(sum(c.isupper() for c in letters) / max(1, len(letters)) / .8),
        _clamp(text.count('\n') / words / .3),
        _clamp(sum(bool(re.match(r'^\s*(?:[-*•]|\d+[.)])\s+', x)) for x in body_lines) / max(1, len(body_lines))),
        _clamp(len(re.findall(r'https?://|www\.', text, re.I)) / 5),
        _clamp(sum(x.lstrip().startswith('>') for x in body_lines) / max(1, len(body_lines))),
        float(bool(REPLY_RE.search(original_body))),
        float(bool(FORWARD_RE.search(original_body))),
        float(bool(zones['greeting'])), float(bool(zones['signature'])),
        float(bool(zones['subject'])), float(bool(zones['body'])),
    )
    return dict(zip(CONTEXT_NAMES, result))

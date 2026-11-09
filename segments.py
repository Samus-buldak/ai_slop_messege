"""MIME parsing and ownership-aware segmentation of email messages.

Replies and forwarded/quoted history are *not* attributed to the current author.
Four zones are kept for compatibility: subject, greeting, body, signature.
"""
from __future__ import annotations

import email
import re
from email import policy
from html.parser import HTMLParser

KINDS = ('subject', 'greeting', 'body', 'signature')
GREETING = re.compile(
    r'^(?:hi\b|hey\b|hello\b|dear\b|good (?:morning|evening|afternoon)\b|'
    r'здравств\w*|добр\w*|привет\w*|хай\b|уважаем\w*|коллеги\b|всем привет\b)', re.I)
SIGNOFF = re.compile(
    r'^(?:--\s*$|с уважением\b|с наилучшими\b|всего доброго\b|'
    r'best regards\b|kind regards\b|warm regards\b|regards\b|sincerely\b|'
    r'cheers\b|best wishes\b|thanks(?:\s+(?:and|&amp;)\s+regards)?\s*[,!]*$|'
    r'отправлено с (?:моего|iphone|ipad|android)\b|sent from my\b)', re.I)
CONTACT = re.compile(r'(?:[\w.+-]+@[\w.-]+\.\w+|\+\d[\d ()-]{6,}|(?:тел|phone|tel)\s*[:.])', re.I)
REPLY_HEADER = re.compile(
    r'^(?:-{2,}\s*(?:original message|forwarded message|исходное сообщение|пересланное сообщение)\s*-{2,}|'
    r'on .{8,200}\bwrote:\s*$|'
    r'в .{8,200}\bписал[аи]?:\s*$|'
    r'from:\s*.+$|от:\s*.+$)', re.I)
DISCLAIMER = re.compile(r'^(?:confidentiality notice|this (?:e-?mail|message) (?:and any attachments )?is confidential|'
                        r'данное (?:электронное )?письмо (?:и вложения )?содержит)', re.I)


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_stack = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = values.get('class', '') or ''
        block_quote = (tag == 'blockquote' or
                       bool(re.search(r'\b(?:gmail_quote|yahoo_quoted|moz-cite-prefix)\b', classes, re.I)) or
                       values.get('type') == 'cite')
        hidden = tag in {'script', 'style', 'head', 'noscript'} or block_quote
        self.skip_stack.append((tag, hidden or self._skipped))
        if not self._skipped and tag in {'p', 'div', 'br', 'li', 'tr', 'h1', 'h2', 'h3'}:
            self.parts.append('\n')

    @property
    def _skipped(self):
        return bool(self.skip_stack and self.skip_stack[-1][1])

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if not self._skipped and tag in {'p', 'div', 'li', 'tr'}:
            self.parts.append('\n')
        # HTML from mailing clients is not always well-formed: recover nesting.
        for i in range(len(self.skip_stack) - 1, -1, -1):
            if self.skip_stack[i][0] == tag:
                del self.skip_stack[i:]
                break

    def handle_data(self, data):
        if not self._skipped:
            self.parts.append(data)


def parse_eml(raw: bytes | str) -> tuple[str, str]:
    if isinstance(raw, str):
        raw = raw.encode('utf-8', 'replace')
    msg = email.message_from_bytes(raw, policy=policy.default)
    subject = str(msg.get('Subject') or '').strip()
    plain, html = [], []
    for part in msg.walk():
        if part.is_multipart() or part.get_content_disposition() == 'attachment':
            continue
        if part.get_content_type() not in {'text/plain', 'text/html'}:
            continue
        try:
            value = part.get_content()
        except (LookupError, UnicodeError, ValueError):
            value = (part.get_payload(decode=True) or b'').decode('utf-8', 'replace')
        if isinstance(value, bytes):
            value = value.decode('utf-8', 'replace')
        (plain if part.get_content_type() == 'text/plain' else html).append(value)
    if plain:
        body = '\n'.join(plain)
    else:
        parser = _HTMLText()
        parser.feed('\n'.join(html))
        body = ''.join(parser.parts)
    body = re.sub(r'\n{3,}', '\n\n', body.replace('\r\n', '\n').replace('\r', '\n')).strip()
    return subject, body


def _remove_quoted_history(lines: list[str]) -> list[str]:
    output = []
    for i, line in enumerate(lines):
        s = line.strip()
        # A reply header only ends owned text if it appears after original content;
        # a standalone "From: ..." at the very beginning may be relevant prose.
        if REPLY_HEADER.match(s) and (i > 0 or s.lower().startswith(('-----', 'on ', 'в '))):
            break
        if DISCLAIMER.match(s) and output:
            break
        if s.startswith('>') or s.startswith('| '):
            continue
        output.append(line)
    return output


def owned_lines(body: str) -> list[str]:
    """Current author's lines, excluding known quoted history and blank edges."""
    lines = [line.rstrip() for line in body.replace('\r\n', '\n').replace('\r', '\n').split('\n')]
    lines = _remove_quoted_history(lines)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def split_email(subject: str, body: str) -> dict[str, str]:
    lines = owned_lines(body)
    greeting = signature = ''
    if lines:
        # Some emails begin with a name / salutation on a single line.
        first = lines[0].strip()
        if len(first) <= 120 and GREETING.match(first):
            greeting = lines.pop(0).strip()
            while lines and not lines[0].strip():
                lines.pop(0)
    # Signature markers near end; prefer the *first* valid marker so name/title remain inside.
    for i in range(max(0, len(lines) - 12), len(lines)):
        if SIGNOFF.match(lines[i].strip()):
            signature = '\n'.join(lines[i:]).strip()
            lines = lines[:i]
            break
    if not signature and len(lines) >= 2:
        blocks = re.split(r'\n\s*\n', '\n'.join(lines))
        last = blocks[-1].strip()
        if len(blocks) > 1 and len(last.splitlines()) <= 5 and CONTACT.search(last):
            signature = last
            lines = '\n\n'.join(blocks[:-1]).split('\n')
    return {'subject': subject.strip(), 'greeting': greeting,
            'body': '\n'.join(lines).strip(), 'signature': signature}


def document_text(zones: dict[str, str]) -> str:
    """Stable embedding input for the current author's text only."""
    labels = {'subject': 'Subject', 'greeting': 'Greeting', 'body': 'Message', 'signature': 'Signature'}
    return '\n'.join(f'{labels[k]}: {zones[k]}' for k in KINDS if zones[k].strip())

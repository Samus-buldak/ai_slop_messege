"""Download public (historical) non-spam mail and prepare provisional human examples.
No API key required. Inspect the output: 'ham' is NOT a guarantee of human authorship.
"""
import argparse
import csv
import hashlib
import html
import io
import random
import re
import tarfile
import urllib.request
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path

URLS = (
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20021010_easy_ham.tar.bz2",
    "https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2",
)
UA = "Kirusha-dataset-preparer/1.0 (educational; SpamAssassin public corpus)"


class StripHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.ignore = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.ignore += 1
        if tag in ('p', 'div', 'li', 'br'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.ignore = max(0, self.ignore - 1)
        if tag in ('p', 'div', 'li'):
            self.parts.append('\n')

    def handle_data(self, value):
        if not self.ignore:
            self.parts.append(value)


def plain_text(raw):
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw)
        part = msg.get_body(preferencelist=('plain', 'html'))
        if part is None:
            part = msg
        data = part.get_content()
        if isinstance(data, bytes):
            data = data.decode('utf-8', 'replace')
        data = str(data)
        if part.get_content_type() == 'text/html':
            stripper = StripHTML()
            stripper.feed(data)
            data = html.unescape(''.join(stripper.parts))
        return str(msg.get('subject', '')).strip(), data
    except Exception:
        return '', ''


def clean(text):
    """Remove obvious addresses, identifiers, quote blocks and mailing-list noise."""
    text = re.sub(r'(?i)https?://\S+|www\.\S+', '', text)
    text = re.sub(r'(?i)\b[\w.+%-]+@[\w.-]+\.[a-z]{2,}\b', '', text)
    text = re.sub(r'(?<!\w)(?:\+?\d[\d .()-]{9,}\d)(?!\w)', '', text)
    lines = []
    for line in text.replace('\r', '').splitlines():
        s = line.strip()
        if s.startswith('>') or s.startswith('|'):
            continue
        if re.match(r'(?i)^on .{10,} wrote:\s*$', s):
            break
        if re.match(r'(?i)^(-{2,}\s*original message|unsubscribe|list-unsubscribe:|x-spam-|message-id:)', s):
            continue
        if re.match(r'^\s*[-_=*~]{8,}\s*$', s):
            continue
        lines.append(s)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(lines)).strip()


def likely_human(subject, body):
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", body)
    if not 35 <= len(words) <= 650:
        return False
    ascii_count = sum(ch.isascii() for ch in body)
    if ascii_count / max(len(body), 1) < 0.82:
        return False
    if len(subject.strip()) < 4:
        return False
    low = (subject + ' ' + body[:600]).lower()
    if any(flag in low for flag in ('out of office', 'automatic reply', 'auto-reply',
                                   'automated message', 'delivery status notification',
                                   'failure notice', 'mail delivery subsystem',
                                   'this is an automatically generated', 'newsletter subscription',
                                   'bugzilla notification', 'cvs commit notification')):
        return False
    return True


def main():
    p = argparse.ArgumentParser(description='Collect provisional human emails from SpamAssassin ham')
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--output', default='data/human_emails.csv')
    p.add_argument('--cache-dir', default='data/download_cache')
    args = p.parse_args()
    if args.count <= 0:
        p.error('--count must be positive')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    items = {}
    for url in URLS:
        local = cache_dir / url.rsplit('/', 1)[-1]
        if not local.exists():
            print('[download]', url, flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=75) as response:
                raw = response.read(15_000_001)
            if len(raw) > 15_000_000:
                raise RuntimeError('Archive unexpectedly large; stopped')
            local.write_bytes(raw)
        print('[read]', local, flush=True)
        with tarfile.open(local, 'r:bz2') as archive:
            for member in archive:
                if not member.isfile() or member.size > 500_000:
                    continue
                infile = archive.extractfile(member)
                if infile is None:
                    continue
                subject, body = plain_text(infile.read())
                subject, body = clean(subject)[:180], clean(body)
                if not likely_human(subject, body):
                    continue
                fingerprint = hashlib.sha256(re.sub(r'\s+', ' ', body).lower().encode('utf-8')).hexdigest()
                if fingerprint not in items:
                    items[fingerprint] = {'subject': subject, 'text': body,
                                          'is_ai': 0, 'source': 'spamassassin_ham_unverified'}
    data = list(items.values())
    rng = random.Random(631)
    rng.shuffle(data)
    if len(data) < args.count:
        raise RuntimeError(f'Only {len(data)} valid candidate emails found (need {args.count}). Adjust filters or sources.')
    selected = data[:args.count]
    with out.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['subject', 'text', 'is_ai', 'source'])
        writer.writeheader()
        writer.writerows(selected)
    print(f'[done] saved {len(selected)} provisional HUMAN samples to {out}')
    print('[!] Inspect samples and remove automated/list messages; HAM != guaranteed human.')
    print('[!] Historical public emails can contain personal data; do not upload raw texts without review.')
    print('[!] Human ham texts are only CANDIDATES; remove auto-mail and reviewed inappropriate items.')


if __name__ == '__main__':
    main()

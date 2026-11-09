"""Produce a reproducible small review sample without sending anything online."""
import argparse
import csv
import random
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--input', default='data/human_emails.csv')
p.add_argument('--output', default='data/review_human_sample.csv')
p.add_argument('--count', type=int, default=50)
args = p.parse_args()
with Path(args.input).open(encoding='utf-8-sig', newline='') as file:
    rows = list(csv.DictReader(file))
selected = random.Random(986).sample(rows, min(len(rows), args.count))
with Path(args.output).open('w', encoding='utf-8-sig', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['subject', 'preview', 'possible_automation', 'review_ok'])
    writer.writeheader()
    for row in selected:
        body = row.get('text','')
        suspicion = bool(re.search(r'(?i)mailing list|unsubscribe|automated|digest|click here|newsletter|notification|commit|announcement', body))
        writer.writerow({'subject':row.get('subject',''), 'preview':re.sub(r'\s+',' ',body)[:450],
                         'possible_automation':int(suspicion), 'review_ok':''})
print(f'[review] Saved {len(selected)} sampled candidate human emails to {args.output}')
print('[!] Manual spot checking does not certify the other emails as human-authored.')

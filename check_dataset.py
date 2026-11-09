"""Inspect the independently gathered HUMAN/AI CSVs before training."""
import argparse
import csv
import hashlib
import re
import statistics
from collections import Counter
from pathlib import Path

COLUMNS = {'subject', 'text', 'is_ai'}


def read(path, expected):
    if not Path(path).exists():
        raise FileNotFoundError(f'Missing: {path}')
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not COLUMNS.issubset(set(reader.fieldnames or [])):
            raise ValueError(f'{path}: missing fields: {COLUMNS - set(reader.fieldnames or [])}')
        for n, row in enumerate(reader, 2):
            if str(row['is_ai']).strip() != str(expected):
                raise ValueError(f'{path} line {n}: wrong is_ai {row["is_ai"]}')
            text = (row['text'] or '').strip()
            if len(text.split()) < 25:
                raise ValueError(f'{path} line {n}: too short ({len(text.split())} words)')
            rows.append(row)
    return rows


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--count', type=int, default=1000)
    a.add_argument('--human', default='data/human_emails.csv')
    a.add_argument('--ai', default='data/ai_generated.csv')
    args = a.parse_args()
    human, ai = read(args.human, 0), read(args.ai, 1)
    print(f'HUMAN candidates: {len(human)}; AI from LLM: {len(ai)}')
    if len(human) < args.count or len(ai) < args.count:
        raise SystemExit('NOT READY: both classes need at least the requested count')
    seen = {}
    for kind, rows in (('HUMAN',human),('AI',ai)):
        words = [len(row['text'].split()) for row in rows]
        subjects = sum(bool(row.get('subject','').strip()) for row in rows)
        digest = [hashlib.sha256(re.sub(r'\s+', ' ',r['text']).lower().encode()).hexdigest() for r in rows]
        print(f'{kind}: median_words={statistics.median(words):.0f}, min={min(words)}, max={max(words)}, subjects={subjects}, unique={len(set(digest))}')
        for h in digest:
            seen.setdefault(h, []).append(kind)
    repeated = sum(len(kinds)>1 for kinds in seen.values())
    across = sum(len(set(kinds))>1 for kinds in seen.values())
    print(f'Exact duplicate contents: {repeated}; across classes: {across}')
    if repeated:
        raise SystemExit('NOT READY: duplicates detected; review and deduplicate before training')
    human_median = statistics.median(len(r['text'].split()) for r in human)
    ai_median = statistics.median(len(r['text'].split()) for r in ai)
    if max(human_median, ai_median) / max(1, min(human_median, ai_median)) > 1.5:
        print('WARNING: class length medians differ strongly; the model may learn length.')
    if any(r.get('source','').startswith('generated:') is False for r in ai):
        print('WARNING: some AI rows do not have confirmed generated: source labels.')
    print('READY for structural training checks. HUMAN ham labels still require manual review!')

if __name__ == '__main__':
    main()

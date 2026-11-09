"""Merge cleaned human and genuine LLM-generated CSVs into train.py input."""
import argparse
import csv
import hashlib
import random
import re
from pathlib import Path


def read_rows(path, expected, count):
    result = []
    seen = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as fh:
        for row in csv.DictReader(fh):
            text = str(row.get('text') or '').strip()
            subject = str(row.get('subject') or '').strip()
            if str(row.get('is_ai')) != str(expected) or not text:
                continue
            key = hashlib.sha256(re.sub(r'\s+', ' ', text).casefold().encode('utf-8')).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            result.append({'subject': subject, 'text': text, 'is_ai': expected})
    if len(result) < count:
        raise RuntimeError(f'{path}: only {len(result)} unique valid rows; need {count}')
    return result[:count]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--human-csv', default='data/human_emails.csv')
    p.add_argument('--ai-csv', default='data/ai_generated.csv')
    p.add_argument('--output', default='data/dataset.csv')
    p.add_argument('--individual', action='store_true', help='Save one CSV for every email as well')
    args = p.parse_args()
    if args.count < 1:
        p.error('--count must be > 0')
    rows_for_human = read_rows(args.human_csv, 0, args.count)
    rows_for_ai = read_rows(args.ai_csv, 1, args.count)
    rows = rows_for_human + rows_for_ai
    random.Random(33).shuffle(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'text', 'is_ai'])
        writer.writeheader()
        writer.writerows(rows)
    if args.individual:
        parent = out.parent / 'individual'
        for name, selected in (('human', rows_for_human), ('ai', rows_for_ai)):
            folder = parent / name
            folder.mkdir(parents=True, exist_ok=True)
            for number, item in enumerate(selected, start=1):
                with (folder / f'{name}_{number:04d}.csv').open('w', encoding='utf-8-sig', newline='') as one:
                    single = csv.DictWriter(one, fieldnames=['subject', 'text', 'is_ai'])
                    single.writeheader()
                    single.writerow(item)
        print(f'[individual] Saved {2 * args.count} separate email CSV files under {parent}')
    print(f'[done] {out}: {2 * args.count} messages ({args.count} human, {args.count} AI)')
    print('[!] Check privacy/labels and evaluate on independent real-world examples.')


if __name__ == '__main__':
    main()

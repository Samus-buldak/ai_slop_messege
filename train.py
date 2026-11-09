"""Four disjoint stages: base embeddings -> stacked trees -> calibration -> held-out test.

Qwen embedding is the default. TF-IDF is optional (--use-tfidf), never required.
Independent trees are trained for no-LLM and with-LLM feature distributions.
"""
import argparse
import asyncio
import csv
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier, export_text

from api import SignalCache
from embedding_pipeline import embedding_vectors, fit_document_projection
from config import (API_KEY, DATA_DIR, MODEL_DIR, SEMANTIC_API_KEY,
                    SEMANTIC_API_URL, SEMANTIC_MODEL)
from scoring import (final_score_v2, full_semantic_coverage, v2_feature_names)
from segments import KINDS, document_text, parse_eml, split_email
from signals_v2 import collect_rows
from segmentation_llm import segment_many


def load_data(csv_path=None):
    if csv_path:
        path = Path(csv_path)
    else:
        path = DATA_DIR / 'dataset.csv'
    rows = []
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            col = next((key for key in ('is_ai', 'ai_assisted', 'label') if key in fields), None)
            if col is None or not fields.intersection({'text', 'body'}):
                raise ValueError('CSV должен содержать text (или body) и is_ai (или ai_assisted / label)')
            if col == 'label' and 'source' in fields:
                print('[!] label может означать СПАМ, не ИИ! Проверь корректность разметки.')
            if col == 'ai_assisted' and 'label' in fields:
                print('[i] выбран ai_assisted как метка ИИ; поле label игнорируется.')
            for line in reader:
                raw_label = str(line.get(col, '')).strip().lower()
                text = str(line.get('text') or line.get('body') or '').strip()
                if not text or raw_label not in {'0', '1', 'true', 'false', 'ai', 'human', 'yes', 'no'}:
                    continue
                label = int(raw_label in {'1', 'true', 'ai', 'yes'})
                rows.append((str(line.get('subject') or '').strip(), text, label))
        print(f'[data] {path}: {len(rows)} строк')
    else:
        if csv_path:
            raise FileNotFoundError(f'Нет файла: {path}')
        for folder, label in (('human', 0), ('ai', 1)):
            for file in sorted((DATA_DIR / folder).rglob('*.eml')):
                subject, body = parse_eml(file.read_bytes())
                if body:
                    rows.append((subject, body, label))
        if not rows:
            raise RuntimeError('Нет размеченных писем. Укажи --csv data/demo_synthetic.csv для '
                               'ДЕМОНСТРАЦИИ или добавь data/dataset.csv / data/human/*.eml + data/ai/*.eml')
    # Убираем точные и очевидные нормализованные дубликаты ДО разделения.
    unique = {}
    contradictory = 0
    for subject, body, label in rows:
        fingerprint = re.sub(r'\s+', ' ', subject + '\n' + body).strip().casefold()
        if fingerprint in unique:
            if unique[fingerprint][2] != label:
                contradictory += 1
                unique.pop(fingerprint)
            continue
        unique[fingerprint] = (subject, body, label)
    rows = list(unique.values())
    if contradictory:
        print(f'[!] найдено {contradictory} противоречивых дубликатов; исключены')
    labels = Counter(x[2] for x in rows)
    print(f'[data] после дедупликации: {len(rows)} (human={labels[0]}, ai={labels[1]})')
    if min(labels[0], labels[1]) < 20:
        raise RuntimeError('Для обучающей/калибровочной/тестовой частей нужно >=20 уникальных '
                           'размеченных писем каждого класса; лучше сотни.')
    return rows


def split_rows(rows, seed, stack_limit, calib_limit, test_limit):
    rng = np.random.default_rng(seed)
    split = {'base': [], 'stack': [], 'calib': [], 'test': []}
    for label in (0, 1):
        idx = np.array([i for i, row in enumerate(rows) if row[2] == label])
        rng.shuffle(idx)
        n = len(idx)
        n_stack = min(stack_limit, max(4, int(round(n * .20))))
        n_calib = min(calib_limit, max(4, int(round(n * .10))))
        n_test = min(test_limit, max(4, int(round(n * .10))))
        if n_stack + n_calib + n_test >= n - 5:
            raise RuntimeError('Слишком мало писем для независимых разделов. Нужно >=20 каждого класса.')
        cursor = 0
        for key, count in (('stack', n_stack), ('calib', n_calib), ('test', n_test)):
            split[key].extend(rows[i] for i in idx[cursor:cursor + count])
            cursor += count
        split['base'].extend(rows[i] for i in idx[cursor:])
    for rows_sub in split.values():
        rng.shuffle(rows_sub)
    return split


def fit_tfidf(rows):
    texts, labels = [], []
    for subject, body, label in rows:
        for text in split_email(subject, body).values():
            if text.strip():
                texts.append(text)
                labels.append(label)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                          max_features=30000, token_pattern=r'(?u)\b\w+\b')
    x = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=350, class_weight='balanced', solver='liblinear', random_state=42)
    clf.fit(x, labels)
    print(f'[tfidf] {len(texts)} зон, {x.shape[1]} признаков')
    return {'vec': vec, 'clf': clf}


def normalize_center(vectors):
    avg = np.asarray(vectors, dtype=float).mean(axis=0)
    norm = np.linalg.norm(avg)
    return (avg / norm).tolist() if norm > 0 else None


def fit_centroids(rows, cache, per_class, llm_segment=False):
    """Per-zone + full-document local Qwen centroids, fit ONLY on base split."""
    if not API_KEY:
        return {'backend': 'none', 'centroids': {}, 'model': None}
    from config import EMBED_MODEL
    probe = embedding_vectors(['embedding probe'], cache=cache)
    if not probe[0]:
        raise RuntimeError('No real /embeddings response. Check LM Studio/Qwen; no fake vectors used.')
    rng = np.random.default_rng(42)
    subset = []
    for label in (0, 1):
        candidates = [row for row in rows if row[2] == label]
        chosen = rng.choice(len(candidates), size=min(per_class, len(candidates)), replace=False)
        subset.extend(candidates[i] for i in chosen)
    if llm_segment:
        enhanced = asyncio.run(segment_many([(a, b) for a, b, _ in subset], cache=cache, verbose=True))
        segs = [z for z, _ in enhanced]
        print(f'[segmentation:base] LLM used: {sum(used for _, used in enhanced)}/{len(enhanced)}')
    else:
        segs = [split_email(subject, body) for subject, body, _ in subset]
    zone_keys = list(dict.fromkeys((kind, zones[kind]) for zones in segs
                                   for kind in KINDS if zones[kind].strip()))
    docs = [document_text(zone) for zone in segs]
    vectors = embedding_vectors([text for _, text in zone_keys] + docs,
                                cache=cache, verbose=True)
    zone_embeddings = dict(zip(zone_keys, vectors[:len(zone_keys)]))
    document_vectors = vectors[len(zone_keys):]
    grouped = {}
    for row, zones, vec in zip(subset, segs, document_vectors):
        if vec is not None:
            grouped.setdefault(('document', row[2]), []).append(vec)
        for kind in KINDS:
            val = zone_embeddings.get((kind, zones[kind]))
            if val is not None:
                grouped.setdefault((kind, row[2]), []).append(val)
    centers = {}
    for (kind, label), vectors in grouped.items():
        if len(vectors) >= 3:
            center = normalize_center(vectors)
            if center is not None:
                centers[f'{kind}_{label}'] = center
    if not all(f'document_{label}' in centers for label in (0, 1)):
        raise RuntimeError('Qwen did not provide enough valid full-document embeddings for two classes')
    print(f'[embedding] {EMBED_MODEL}: {len(centers)} centroids including document; dim={len(probe[0])}')
    projection = fit_document_projection(document_vectors)
    return {'backend': 'api', 'centroids': centers, 'model': EMBED_MODEL,
            'dimension': len(probe[0]), 'document_format': 'document_text_v1',
            'projection': projection}


def score_rows(rows, tfidf, centroids, cache, use_sem, llm_segment=False):
    """Legacy helper name kept for tests / callers, using the v2 feature engine."""
    enhanced = (asyncio.run(segment_many([(a, b) for a, b, _ in rows], cache=cache, verbose=True))
                if llm_segment else None)
    if enhanced is not None:
        print(f'[segmentation] LLM used: {sum(ok for _, ok in enhanced)}/{len(enhanced)}')
    return collect_rows(rows, tfidf, centroids, cache, with_llm=use_sem,
                        verbose=True, zone_overrides=[z for z, _ in enhanced] if enhanced else None,
                        use_owned_context=llm_segment)


def best_cut(labels, scores):
    """O(N log N): реальные границы между отсортированными score, максимум Youden J."""
    pairs = sorted(zip(scores, labels))
    humans = sum(label == 0 for _, label in pairs)
    ais = len(pairs) - humans
    if not humans or not ais:
        return None, None
    distinct = sorted(set(score for score, _ in pairs))
    candidates = ([max(0., distinct[0] - 1e-6)] +
                  [(a + b) / 2 for a, b in zip(distinct[:-1], distinct[1:])] +
                  [min(1., distinct[-1] + 1e-6)])
    values = []
    for cut in candidates:
        tp = sum(s >= cut and y == 1 for s, y in pairs) / ais
        fp = sum(s >= cut and y == 0 for s, y in pairs) / humans
        values.append((tp - fp, cut))
    best = max(x for x, _ in values)
    threshold = float(np.median([cut for x, cut in values if abs(x - best) < 1e-10]))
    return round(threshold, 5), round(best, 4)


def calibrate_segments(rows):
    thresholds = {}
    for kind in KINDS:
        thresholds[kind] = {}
        for mode in ('no_llm', 'with_llm'):
            subset = [(r['label'], r['scores'][kind][mode]) for r in rows
                      if r['scores'][kind]['present'] and
                      r['scores'][kind][mode] is not None and
                      (mode == 'no_llm' or r['scores'][kind]['semantic_llm'] is not None)]
            count = Counter(y for y, _ in subset)
            if min(count[0], count[1]) >= 3:
                threshold, youden = best_cut([y for y, _ in subset], [s for _, s in subset])
                thresholds[kind][mode] = threshold
                thresholds[kind][mode + '_n'] = len(subset)
                thresholds[kind][mode + '_youden'] = youden
            else:
                thresholds[kind][mode] = None
    return thresholds


def engine_thresholds(calib, tree, mode):
    eligible = [r for r in calib if mode == 'no_llm' or full_semantic_coverage(r['scores'])]
    if len({r['label'] for r in eligible}) < 2 or min(Counter(r['label'] for r in eligible).values()) < 3:
        return None
    probs = tree.predict_proba(np.array([r['features_' + mode] for r in eligible]))[:, 1]
    labels = [r['label'] for r in eligible]
    finals = [final_score_v2(r['scores'], p, mode) for r, p in zip(eligible, probs)]
    split, j = best_cut(labels, finals)
    human = float(np.quantile([s for s, y in zip(finals, labels) if y == 0], .90))
    ai = float(np.quantile([s for s, y in zip(finals, labels) if y == 1], .10))
    if human < ai:
        gap = min(.15, max(.025, .15 * (ai - human)))
        lower, upper = max(human, split - gap), min(ai, split + gap)
        if lower < upper:
            human, ai = lower, upper
        else:
            human, ai = max(0., split - .025), min(1., split + .025)
    else:
        human, ai = max(0., split - .025), min(1., split + .025)
    return {'human_cut': round(human, 5), 'ai_cut': round(ai, 5),
            'youden_cut': split, 'youden_j': j, 'n_calibration': len(eligible)}


def verdict(score, thresholds):
    return ('HUMAN' if score < thresholds['human_cut'] else
            'AI' if score >= thresholds['ai_cut'] else 'UNCERTAIN')


def evaluate(rows, trees, engine):
    report = {}
    for mode, tree in trees.items():
        cut = engine.get(mode)
        if not cut:
            continue
        eligible = [r for r in rows if mode == 'no_llm' or full_semantic_coverage(r['scores'])]
        if not eligible or len({r['label'] for r in eligible}) < 2:
            continue
        probs = tree.predict_proba(np.asarray([r['features_' + mode] for r in eligible]))[:, 1]
        finals = [final_score_v2(r['scores'], p, mode) for r, p in zip(eligible, probs)]
        y = [r['label'] for r in eligible]
        preds = [int(s >= cut['youden_cut']) for s in finals]
        decisions = [verdict(s, cut) for s in finals]
        report[mode] = {
            'n_test': len(y),
            'roc_auc': round(float(roc_auc_score(y, finals)), 4),
            'balanced_accuracy_at_youden': round(float(balanced_accuracy_score(y, preds)), 4),
            'precision_at_youden': round(float(precision_score(y, preds, zero_division=0)), 4),
            'recall_at_youden': round(float(recall_score(y, preds, zero_division=0)), 4),
            'uncertain_fraction': round(decisions.count('UNCERTAIN') / len(decisions), 4),
            'human_wrongly_flagged_ai': sum(z == 'AI' and label == 0 for z, label in zip(decisions, y)),
            'ai_wrongly_flagged_human': sum(z == 'HUMAN' and label == 1 for z, label in zip(decisions, y)),
            'confusion_at_youden': confusion_matrix(y, preds, labels=[0, 1]).tolist(),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='CSV; default data/dataset.csv')
    parser.add_argument('--offline', '--no-llm', action='store_true',
                        help='disable all API calls; requires --use-tfidf')
    parser.add_argument('--no-semantic', action='store_true', help='alias for default: do not use a chat LLM')
    parser.add_argument('--llm-segmentation', action='store_true',
                        help='LLM segments own lines in every training/evaluation stage (requires --with-llm)')
    parser.add_argument('--with-llm', action='store_true',
                        help='include independently scored chat-LLM features in a second tree')
    parser.add_argument('--use-tfidf', action='store_true',
                        help='optional classic baseline; by default Qwen-only')
    parser.add_argument('--centroid-per-class', type=int, default=120)
    parser.add_argument('--stack-per-class', type=int, default=150)
    parser.add_argument('--calib-per-class', type=int, default=120)
    parser.add_argument('--test-per-class', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.offline and not args.use_tfidf:
        parser.error('--offline requires --use-tfidf, otherwise neither text model nor embeddings are available')
    if args.llm_segmentation and not args.with_llm:
        parser.error('--llm-segmentation requires --with-llm')
    if args.llm_segmentation and args.use_tfidf:
        parser.error('--llm-segmentation and --use-tfidf cannot be combined yet')
    if args.with_llm and (args.no_semantic or args.offline):
        parser.error('--with-llm cannot be combined with --no-semantic / --offline')
    if not args.offline and not API_KEY and not args.use_tfidf:
        parser.error('Qwen endpoint is not configured; use local_qwen.py train or --use-tfidf --offline')
    t0 = time.perf_counter()
    rows = load_data(args.csv)
    parts = split_rows(rows, args.seed, args.stack_per_class, args.calib_per_class, args.test_per_class)
    print('[split]', {k: dict(Counter(row[2] for row in v)) for k, v in parts.items()})
    tfidf = fit_tfidf(parts['base']) if args.use_tfidf else None
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    use_api = bool(API_KEY) and not args.offline
    use_sem = bool(SEMANTIC_API_KEY) and args.with_llm
    if args.with_llm and not use_sem:
        parser.error('No chat LLM API key configured for --with-llm')
    with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
        centroids = fit_centroids(parts['base'], cache, max(3, args.centroid_per_class), llm_segment=args.llm_segmentation) if use_api else {
            'backend': 'none', 'centroids': {}, 'model': None}
        print('[stack] fitting no-LLM tree on stack-only examples')
        stack = score_rows(parts['stack'], tfidf, centroids, cache, use_sem=use_sem, llm_segment=args.llm_segmentation)
        x_stack = np.asarray([r['features_no_llm'] for r in stack])
        y_stack = np.asarray([r['label'] for r in stack])
        def new_tree():
            return DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(5, len(stack) // 50),
                                          class_weight='balanced', random_state=args.seed)
        trees = {'no_llm': new_tree().fit(x_stack, y_stack)}
        print(export_text(trees['no_llm'], feature_names=v2_feature_names()))
        if use_sem:
            sem_stack = [r for r in stack if full_semantic_coverage(r['scores'])]
            if len({r['label'] for r in sem_stack}) < 2 or len(sem_stack) < 12:
                raise RuntimeError('Too few complete chat LLM judgments for with_llm tree. '
                                   'Load a local chat model or omit --with-llm.')
            trees['with_llm'] = new_tree().fit(
                np.asarray([r['features_with_llm'] for r in sem_stack]),
                np.asarray([r['label'] for r in sem_stack]))
            print('[stack:with_llm] trained on', len(sem_stack), 'complete judgments')
        print('[calib] choosing cutoffs on independent samples')
        calib = score_rows(parts['calib'], tfidf, centroids, cache, use_sem=use_sem, llm_segment=args.llm_segmentation)
        segment_cuts = calibrate_segments(calib)
        engine = {'no_llm': engine_thresholds(calib, trees['no_llm'], 'no_llm')}
        if engine['no_llm'] is None:
            raise RuntimeError('Cannot calibrate a no-LLM decision threshold')
        if 'with_llm' in trees:
            engine['with_llm'] = engine_thresholds(calib, trees['with_llm'], 'with_llm')
            if engine['with_llm'] is None:
                raise RuntimeError('Cannot calibrate a with-LLM decision threshold')
        print('[thresholds] segments:', json.dumps(segment_cuts, ensure_ascii=False))
        print('[thresholds] engine:', engine)
        print('[test] held-out data, never used in fitting / calibration')
        test = score_rows(parts['test'], tfidf, centroids, cache, use_sem=use_sem, llm_segment=args.llm_segmentation)
    report = evaluate(test, trees, engine)
    print('[metrics] held-out:', json.dumps(report, indent=2, ensure_ascii=False))
    thresholds = {'schema_version': 2, 'segments': segment_cuts, 'engine': engine,
                  'embedding_backend': centroids['backend'],
                  'use_tfidf': args.use_tfidf, 'with_llm_trained': bool('with_llm' in trees),
                  'llm_segmentation': bool(args.llm_segmentation),
                  'segmentation_version': ('line-boundaries-v1' if args.llm_segmentation else 'rules'),
                  'llm_model': (SEMANTIC_MODEL if use_sem else None),
                  'llm_endpoint': (SEMANTIC_API_URL if use_sem else None),
                  'feature_names': v2_feature_names(),
                  'class_mapping': {'0': 'HUMAN', '1': 'AI'}}
    # Atomic writes prevent a partial, mismatched model directory after interruption.
    for name, artifact in (('tfidf.pkl', tfidf), ('tree.pkl', trees)):
        tmp = MODEL_DIR / (name + '.tmp')
        with tmp.open('wb') as fh:
            pickle.dump(artifact, fh)
        tmp.replace(MODEL_DIR / name)
    for name, obj in (('centroids.json', centroids), ('thresholds.json', thresholds),
                      ('test_metrics.json', report)):
        tmp = MODEL_DIR / (name + '.tmp')
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(MODEL_DIR / name)
    print(f'[done] {time.perf_counter() - t0:.1f} s; models and metrics in {MODEL_DIR}')
    print('[!] Synthetic / source-shifted corpus is NOT evidence of robust real-world AI detection.')


if __name__ == '__main__':
    main()

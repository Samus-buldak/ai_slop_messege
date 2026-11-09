"""Analyze emails with v2 Qwen+context+LLM decision trees (v1 backward compatible)."""
import argparse
import asyncio
import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from api import SignalCache, api_signals
from config import (API_KEY, MODEL_DIR, EMBED_MODEL, MODEL,
                    SEMANTIC_API_KEY, SEMANTIC_API_URL, SEMANTIC_MODEL)
from scoring import (embedding_score, features, final_score, final_score_v2,
                     full_semantic_coverage, scores_for_email, v2_feature_names)
from segments import KINDS, parse_eml, split_email
from signals_v2 import collect_rows
from segmentation_llm import segment_many
from media_images import extract_images
from vision import assess_images, combine_scores, IMAGE_WEIGHT

RU = {'subject': 'тема', 'greeting': 'приветствие',
      'body': 'тело', 'signature': 'подпись'}


@lru_cache(maxsize=1)
def load_model():
    names = ('tfidf.pkl', 'tree.pkl', 'thresholds.json', 'centroids.json')
    missing = [name for name in names if not (MODEL_DIR / name).exists()]
    if missing:
        raise RuntimeError('Не найдены модели: ' + ', '.join(missing) + '. Сначала: python train.py')
    # Никогда не загружать чужой/непроверенный pickle: Python pickle может выполнять код.
    with (MODEL_DIR / 'tfidf.pkl').open('rb') as f:
        tfidf = pickle.load(f)
    with (MODEL_DIR / 'tree.pkl').open('rb') as f:
        tree = pickle.load(f)
    thresholds = json.loads((MODEL_DIR / 'thresholds.json').read_text(encoding='utf-8'))
    centers = json.loads((MODEL_DIR / 'centroids.json').read_text(encoding='utf-8'))
    return tfidf, tree, thresholds, centers


def _score_email_v1(raw: bytes | str):
    tfidf, tree, thresholds, centroids = load_model()
    subject, body = parse_eml(raw)
    segs = split_email(subject, body)
    present = [(k, segs[k]) for k in KINDS if segs[k].strip()]
    if not present:
        return {'subject': subject, 'verdict': 'UNCERTAIN', 'reason': 'empty_message',
                'segments': {k: {'present': False, 'score': None} for k in KINDS},
                'final_score': None, 'flagged_segments': []}
    texts = [text for _, text in present]
    predictions = tfidf['clf'].predict_proba(tfidf['vec'].transform(texts))[:, 1]
    tf_scores = dict(zip((k for k, _ in present), map(float, predictions)))

    wants_embedding = centroids['backend'] == 'api'
    # API вызовы запускаются параллельно. HTTP-сервер вызывает score_email в threadpool.
    if API_KEY:
        with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
            embed_api, sem_api = asyncio.run(api_signals([segs], want_embedding=wants_embedding,
                                                          want_semantic=bool(thresholds['engine'].get('with_llm')),
                                                          cache=cache))
    else:
        embed_api, sem_api = {}, {}
    em_scores, sem_scores = {}, {}
    for kind, text in present:
        c = centroids['centroids']
        em_scores[kind] = embedding_score(embed_api.get((kind, text)),
                                          c.get(f'{kind}_1'), c.get(f'{kind}_0'))
        sem_scores[kind] = sem_api.get((kind, text))
    scores = scores_for_email(segs, tf_scores, em_scores, sem_scores)
    model_features = np.asarray([features(scores)])
    tree_p = float(tree.predict_proba(model_features)[0][1])

    engine = thresholds['engine']
    mode = 'with_llm' if engine.get('with_llm') and full_semantic_coverage(scores) else 'no_llm'
    final = final_score(scores, tree_p, mode)
    cuts = engine[mode]
    warnings = []
    # Независимый fallback после отказа LLM используется только в обученном no_llm-режиме.
    if centroids['backend'] == 'api' and any(
        scores[k]['embedding'] is None and
        centroids['centroids'].get(f'{k}_1') is not None and
        centroids['centroids'].get(f'{k}_0') is not None for k, _ in present
    ):
        warnings.append('API эмбеддингов недоступен: распределение признаков изменилось. Ручная проверка.')
    if API_KEY and mode == 'no_llm':
        warnings.append('Семантика API недоступна или не калибрована: использован no_llm режим.')
    verdict = ('HUMAN' if final < cuts['human_cut'] else
               'AI' if final >= cuts['ai_cut'] else 'UNCERTAIN')
    if warnings and centroids['backend'] == 'api' and any(s['embedding'] is None for s in scores.values() if s['present']):
        verdict = 'UNCERTAIN'

    out = {}
    flagged = []
    for kind in KINDS:
        item = scores[kind]
        if not item['present']:
            out[kind] = {'present': False, 'score': None}
            continue
        seg_mode = mode if mode == 'no_llm' else (
            'with_llm' if item['semantic_llm'] is not None else 'no_llm')
        threshold = thresholds['segments'][kind].get(seg_mode)
        value = item[seg_mode]
        label = ('AI' if threshold is not None and value >= threshold else
                 'HUMAN' if threshold is not None else 'UNDETERMINED')
        if label == 'AI':
            flagged.append(kind)
        out[kind] = {'present': True, 'text_preview': segs[kind][:300],
                     'tfidf': item['tfidf'], 'embedding': item['embedding'],
                     'semantic_llm': item['semantic_llm'], 'score': value,
                     'threshold': threshold, 'label': label}

    certain_labels = [v['label'] for v in out.values() if v.get('label') in {'AI', 'HUMAN'}]
    if 'AI' in certain_labels and 'HUMAN' in certain_labels and verdict != 'UNCERTAIN':
        verdict = 'UNCERTAIN'
        warnings.append('У сегментов конфликтующие метки: возможно смешанное авторство.')

    return {'subject': subject, 'segments': out, 'mode': mode,
            'tree_probability': round(tree_p, 5), 'final_score': round(float(final), 5),
            'engine_thresholds': cuts, 'verdict': verdict,
            'flagged_segments': flagged, 'warnings': warnings,
            'api_key_set': bool(API_KEY), 'embedding_backend': centroids['backend'],
            'note': 'Scores/пороги не являются достоверной вероятностью авторства ИИ.'}


def _tree_path(tree, row):
    """Human-readable actual split decisions; not a causal explanation."""
    names = v2_feature_names()
    path = []
    node = 0
    while tree.tree_.children_left[node] != -1:
        index = int(tree.tree_.feature[node])
        threshold = float(tree.tree_.threshold[node])
        value = float(row[index])
        branch = 'left' if value <= threshold else 'right'
        path.append({'feature': names[index], 'value': round(value, 4),
                     'operator': '<=' if branch == 'left' else '>',
                     'threshold': round(threshold, 4)})
        node = (tree.tree_.children_left[node] if branch == 'left'
                else tree.tree_.children_right[node])
    return path


def _score_email_v2(raw: bytes | str, model):
    tfidf, trees, thresholds, centroids = model
    if centroids['backend'] == 'api' and centroids.get('model') != EMBED_MODEL:
        raise RuntimeError('Qwen embedding model changed: retrain the detector before analysis')
    subject, body = parse_eml(raw)
    requires_llm_segmentation = thresholds.get('llm_segmentation', False)
    segmentation_ok = not requires_llm_segmentation
    if requires_llm_segmentation:
        compatible = (bool(SEMANTIC_API_KEY) and
                      thresholds.get('llm_model') == SEMANTIC_MODEL and
                      thresholds.get('llm_endpoint', SEMANTIC_API_URL) == SEMANTIC_API_URL)
        if compatible:
            with SignalCache(MODEL_DIR / 'cache.sqlite3') as seg_cache:
                segments_checked = asyncio.run(segment_many([(subject, body)], cache=seg_cache))
            zones, segmentation_ok = segments_checked[0]
        else:
            zones = split_email(subject, body)
    else:
        zones = split_email(subject, body)
    present = [k for k in KINDS if zones[k].strip()]
    if not present:
        return {'subject': subject, 'verdict': 'UNCERTAIN', 'reason': 'empty_message',
                'segments': {k: {'present': False, 'score': None} for k in KINDS},
                'final_score': None, 'flagged_segments': [], 'warnings': []}
    chat_compatible = ('with_llm' in trees and bool(SEMANTIC_API_KEY) and
                       thresholds.get('llm_model') == SEMANTIC_MODEL and
                       thresholds.get('llm_endpoint', SEMANTIC_API_URL) == SEMANTIC_API_URL)
    with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
        rows = collect_rows([(subject, body, 0)], tfidf, centroids, cache,
                            with_llm=bool(chat_compatible), zone_overrides=[zones],
                            use_owned_context=requires_llm_segmentation)
    item = rows[0]
    scores = item['scores']
    mode = ('with_llm' if 'with_llm' in trees and
            full_semantic_coverage(scores) else 'no_llm')
    tree = trees[mode]
    feature_vector = item['features_' + mode]
    p = float(tree.predict_proba(np.asarray([feature_vector]))[0, 1])
    final = final_score_v2(scores, p, mode)
    cut = thresholds['engine'][mode]
    label = ('HUMAN' if final < cut['human_cut'] else
             'AI' if final >= cut['ai_cut'] else 'UNCERTAIN')
    warnings = []
    if centroids['backend'] == 'api' and item['doc_signals']['document_embedding'] is None:
        warnings.append('Qwen embedding недоступен или размерность вектора не совпадает. Требуется ручная проверка.')
        label = 'UNCERTAIN'
    if 'with_llm' in trees and mode == 'no_llm':
        warnings.append('Разговорная LLM не настроена, модель/endpoint не совпадают или не все части оценены: используется no_llm.')
    if requires_llm_segmentation and not segmentation_ok:
        warnings.append('LLM-сегментация недоступна; использован алгоритм по правилам. Модель обучена на другой сегментации.')
        label = 'UNCERTAIN'
    flagged = []
    details = {}
    for kind in KINDS:
        s = scores[kind]
        if not s['present']:
            details[kind] = {'present': False, 'score': None}
            continue
        selected = mode
        if selected == 'with_llm' and s['semantic_llm'] is None:
            selected = 'no_llm'
        threshold = thresholds['segments'][kind].get(selected)
        value = s[selected]
        if threshold is None or value is None:
            seg_label = 'UNDETERMINED'
        else:
            seg_label = 'AI' if value >= threshold else 'HUMAN'
        if seg_label == 'AI':
            flagged.append(kind)
        details[kind] = {'present': True, 'text_preview': zones[kind][:300],
                         'tfidf': s['tfidf'], 'embedding': s['embedding'],
                         'embedding_ai_similarity': s['embedding_ai_similarity'],
                         'embedding_human_similarity': s['embedding_human_similarity'],
                         'semantic_llm': s['semantic_llm'], 'score': value,
                         'threshold': threshold, 'label': seg_label}
    confident_labels = {v['label'] for v in details.values()
                        if v.get('label') in ('AI', 'HUMAN')}
    if len(confident_labels) == 2 and label != 'UNCERTAIN':
        label = 'UNCERTAIN'
        warnings.append('Части письма противоречат друг другу: возможно смешанное авторство.')
    return {'subject': subject, 'segments': details,
            'mode': mode, 'schema_version': 2,
            'segmentation_method': ('llm_lines' if requires_llm_segmentation and segmentation_ok else 'rules'),
            'embedding_backend': centroids['backend'], 'embedding_model': centroids['model'],
            'use_tfidf': tfidf is not None,
            'context_features': item['context'],
            'document_embedding_features': item['doc_signals'],
            'tree_probability': round(p, 5),
            'tree_decision_path': _tree_path(tree, feature_vector),
            'final_score': round(float(final), 5), 'engine_thresholds': cut,
            'verdict': label, 'flagged_segments': flagged, 'warnings': warnings,
            'api_key_set': bool(API_KEY),
            'note': 'Scores are heuristic; they cannot establish AI authorship with certainty.'}


def score_email(raw: bytes | str):
    """Text scoring + optional image judgment; no vision API call for text-only mail."""
    model = load_model()
    if model[2].get('schema_version') == 2:
        if not isinstance(model[1], dict) or 'no_llm' not in model[1]:
            raise RuntimeError('Incompatible v2 tree.pkl, retrain with the updated train.py')
        result = _score_email_v2(raw, model)
    else:
        result = _score_email_v1(raw)
    photos, media_warnings = extract_images(raw)
    result.setdefault('warnings', []).extend(media_warnings)
    result['text_final_score'] = result.get('final_score')
    result['text_verdict'] = result.get('verdict')
    result['photos'] = {'count': len(photos), 'checked': 0, 'image_score': None,
                        'weight': IMAGE_WEIGHT if photos else 0., 'items': [],
                        'fusion_calibrated': False}
    if not photos:
        return result
    with SignalCache(MODEL_DIR / 'cache.sqlite3') as cache:
        judgments = asyncio.run(assess_images(photos, cache=cache))
    scores = []
    for photo, judgment in zip(photos, judgments):
        value = judgment['score']
        if value is not None:
            scores.append(value)
        result['photos']['items'].append({
            'filename': photo.filename, 'width': photo.width, 'height': photo.height,
            'score': round(value, 5) if value is not None else None,
            'reason': judgment['reason']})
    result['photos']['checked'] = len(scores)
    if not scores:
        result['warnings'].append('Изображения найдены, но vision-LLM не смогла их оценить; итог по тексту не изменён.')
        return result
    image_score = sum(scores) / len(scores)
    result['photos']['image_score'] = round(image_score, 5)
    text_score = result['text_final_score']
    if text_score is None:
        result['warnings'].append('Текст отсутствует: изображение оценено, но итоговое авторство сообщения определить нельзя.')
        return result
    combined = combine_scores(text_score, scores)
    result['final_score'] = round(combined, 5)
    result['warnings'].append('Оценка изображений — эвристика vision-LLM, вес 20%; итог НЕ откалиброван на фото.')
    cuts = result.get('engine_thresholds')
    if cuts and result.get('verdict') != 'UNCERTAIN':
        predicted = ('HUMAN' if combined < cuts['human_cut'] else
                     'AI' if combined >= cuts['ai_cut'] else 'UNCERTAIN')
        if abs(text_score - image_score) >= 0.55:
            result['verdict'] = 'UNCERTAIN'
            result['warnings'].append('Текст и фото существенно противоречат друг другу.')
        else:
            result['verdict'] = predicted
    return result


def fmt(value):
    return f'{value:.3f}' if value is not None else '   — '


def main():
    parser = argparse.ArgumentParser(description='Детектор AI-авторства .eml')
    parser.add_argument('file', help='путь к письму .eml или - для stdin')
    parser.add_argument('--json', action='store_true', help='машиночитаемый JSON')
    args = parser.parse_args()
    raw = sys.stdin.buffer.read() if args.file == '-' else Path(args.file).read_bytes()
    result = score_email(raw)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f'Тема: {result["subject"] or "(нет)"}')
    print(f'Режим: {result.get("mode", "нет")}; embeddings: {result.get("embedding_backend", "нет")}')
    print(f'{"СЕГМЕНТ":<16} {"TF-IDF":>9} {"EMBED":>9} {"LLM":>9} {"SCORE":>9} {"ПОРОГ":>9} МЕТКА')
    for k in KINDS:
        v = result['segments'][k]
        if not v['present']:
            print(f'{RU[k]:<16} (зона отсутствует)')
            continue
        print(f'{RU[k]:<16} {fmt(v["tfidf"]):>9} {fmt(v["embedding"]):>9} '
              f'{fmt(v["semantic_llm"]):>9} {fmt(v["score"]):>9} '
              f'{fmt(v["threshold"]):>9} {v["label"]}')
    print(f'Decision Tree: {fmt(result.get("tree_probability"))}')
    print(f'Final score: {fmt(result.get("final_score"))}')
    if result.get('photos', {}).get('count'):
        print('Изображений:', result['photos']['count'], 'оценено:', result['photos']['checked'],
              'image_score:', fmt(result['photos']['image_score']))
    print(f'ВЕРДИКТ: {result["verdict"]}')
    for warning in result.get('warnings', []):
        print('[!] ' + warning)
    print('Оценка вероятностная и не доказывает реальное авторство.')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        sys.exit(f'Ошибка: {exc}')

"""Generate real LLM-authored email samples on the user's computer via their own API.
Reads API_KEY / API_URL / MODEL from project's config.py, NEVER prints/stores API_KEY.
Batching, bounded concurrency, retries and append-only checkpoints.
"""
import argparse
import asyncio
import csv
import json
import random
import re
from pathlib import Path

import httpx
from config import API_KEY, API_URL, MODEL

TOPICS = [
    'changing a meeting time', 'clarifying a previous request', 'project updates',
    'asking for help with a technical issue', 'planning a visit', 'sending a reminder',
    'asking a colleague a question', 'inviting a friend', 'following up on a purchase',
    'explaining a delay', 'discussing classwork', 'coordinating travel',
    'responding to a complaint', 'sharing event details', 'asking for feedback',
    'requesting a document', 'organizing a group activity', 'reporting a small problem',
    'apologizing for a mix-up', 'giving informal advice', 'rescheduling a call',
    'asking for clarification about deadlines', 'replying about an appointment',
    'thanking a teammate', 'describing a weekend plan', 'chatting about hobbies',
    'arranging a delivery', 'requesting information about a service',
]
STYLES = [
    'warm and informal', 'direct and practical', 'somewhat rushed',
    'polite and professional', 'very short sentences', 'slightly disorganized',
    'friendly with colloquialisms', 'matter-of-fact', 'chatty', 'reserved',
    'casual but respectful', 'straight to the point',
]
ROLES = ['student', 'colleague', 'team lead', 'customer', 'classmate', 'neighbor', 'family friend', 'volunteer', 'organizer', 'support agent', 'club member', 'freelancer', 'community member']
RECIPIENTS = ['a close friend', 'a professor', 'a coworker', 'a small group', 'a new client', 'an acquaintance', 'a teammate', 'a service provider', 'a project manager', 'a neighbor']
DETAILS = ['a last-minute complication', 'a misread calendar entry', 'a missing attachment', 'a slight disagreement', 'a suggested alternative', 'a tight deadline', 'an unexpectedly good outcome', 'a practical concern', 'an update after an earlier reply', 'a small favor', 'an unclear instruction', 'a conflicting commitment', 'a follow-up after two days', 'a time-sensitive change', 'a helpful suggestion', 'an unfinished task', 'a travel disruption', 'a change in priorities', 'an unanticipated expense', 'a small technical issue']
FORMATS = [
    'with a greeting but no signature', 'with a short sign-off', 'without a greeting',
    'with ordinary paragraph breaks', 'as a reply to an earlier conversation',
    'with a subject and a single paragraph',
]


def build_requests(total, human_path, lang):
    human = []
    path = Path(human_path)
    if path.exists():
        with path.open('r', encoding='utf-8-sig', newline='') as file:
            human = list(csv.DictReader(file))
    if human:
        lengths = [len(re.findall(r'\S+', row.get('text', ''))) for row in human]
        lengths = [min(max(x, 45), 250) for x in lengths if x > 0]
    else:
        lengths = [70, 90, 110, 130, 160, 200]
    rng = random.Random(97831)
    tasks = []
    for i in range(1, total + 1):
        tasks.append({
            'i': i,
            'topic': rng.choice(TOPICS),
            'style': rng.choice(STYLES),
            'format': rng.choice(FORMATS),
            'words': rng.choice(lengths),
            'sender': rng.choice(ROLES),
            'recipient': rng.choice(RECIPIENTS),
            'detail': rng.choice(DETAILS),
            'lang': 'Russian' if lang == 'ru' else 'English',
        })
    return tasks


def parse_output(content, tasks):
    if isinstance(content, list):
        content = ''.join(str(c.get('text', '')) for c in content if isinstance(c, dict))
    if not isinstance(content, str):
        raise ValueError('Non-text LLM response')
    content = content.strip()
    if content.startswith('```'):
        content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content, flags=re.I).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find('[')
        if start < 0:
            raise ValueError('No JSON array in LLM response')
        parsed, _ = json.JSONDecoder().raw_decode(content[start:])
    if isinstance(parsed, dict):
        parsed = parsed.get('emails')
    if not isinstance(parsed, list):
        raise ValueError('LLM did not return an array')
    by_id = {}
    for row in parsed:
        if not isinstance(row, dict):
            continue
        try:
            i = int(row['i'])
            subject = str(row.get('subject', '')).strip()
            body = str(row.get('text') or row.get('body') or '').strip()
        except (ValueError, TypeError, KeyError):
            continue
        if subject and 25 <= len(body.split()) <= 500 and i not in by_id:
            by_id[i] = {'id': i, 'subject': subject[:180], 'text': body,
                        'is_ai': 1, 'source': 'generated:' + MODEL}
    expected = {task['i'] for task in tasks}
    if set(by_id) != expected:
        raise ValueError(f'Expected {len(expected)} emails, got {len(expected & set(by_id))}')
    return [by_id[task['i']] for task in tasks]


async def generate_batch(client, tasks, retries):
    instr = (
        'You are writing *fictional* email examples for a research dataset. '
        'Write the emails yourself. Produce a JSON ARRAY with one object for every requested item. '
        'Each object: {"i": integer, "subject": string, "text": string}. '
        'Use exactly the provided IDs and requested language. Do not mention AI, datasets, or prompts. '
        'Do not copy existing real emails. Vary sentence structure and wording. '
        'No real names, email addresses, phone numbers, or private information. '
        'Write natural email prose (not lists of instructions). '
        'Each email should present a specific believable situation and not be a template. '
        'Do not mimic a specific real person or historical source. '
        'Keep greetings and sign-offs only if appropriate for the format, and avoid repetitive phrases. '
        'Word counts can deviate by up to 30%. Output JSON only.'
    )
    payload = {
        'model': MODEL,
        'messages': [
            {'role': 'system', 'content': instr},
            {'role': 'user', 'content': json.dumps(tasks, ensure_ascii=False)},
        ],
    }
    for attempt in range(retries + 1):
        try:
            resp = await client.post(API_URL + '/chat/completions', json=payload,
                                     headers={'Authorization': 'Bearer ' + API_KEY})
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError('temporary API error', request=resp.request, response=resp)
            if resp.status_code >= 400:
                snippet = resp.text[:200].replace('\n', ' ')
                raise RuntimeError(f'API returned HTTP {resp.status_code}: {snippet}')
            message = resp.json()['choices'][0]['message']['content']
            return parse_output(message, tasks)
        except RuntimeError:
            raise
        except (httpx.TransportError, httpx.HTTPStatusError, ValueError, KeyError, IndexError, json.JSONDecodeError) as e:
            if attempt == retries:
                raise RuntimeError(f'Generation failed after {retries + 1} attempts: {type(e).__name__}: {e}') from e
            await asyncio.sleep(min(10, 0.7 * 2 ** attempt))


async def generate_with_fallback(client, tasks, retries):
    """Retry smaller chunks if multi-item JSON breaks; never synthesize labels locally."""
    try:
        return await generate_batch(client, tasks, retries)
    except RuntimeError:
        if len(tasks) == 1:
            raise
        mid = len(tasks) // 2
        left = await generate_with_fallback(client, tasks[:mid], retries)
        right = await generate_with_fallback(client, tasks[mid:], retries)
        return left + right


def previous_ids(path):
    if not path.exists():
        return set()
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        return {int(row['id']) for row in csv.DictReader(file) if row.get('id', '').isdigit()}


async def run(args):
    if not API_KEY:
        raise RuntimeError('API_KEY is missing; configure your local .env first.')
    if args.count < 1 or args.batch < 1 or args.parallel < 1:
        raise ValueError('count, batch and parallel must be > 0')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = previous_ids(out)
    plan = Path(args.plan)
    plan.parent.mkdir(parents=True, exist_ok=True)
    if plan.is_file():
        with plan.open(encoding='utf-8-sig', newline='') as file:
            all_tasks = [dict(row) for row in csv.DictReader(file)]
        if len(all_tasks) < args.count:
            old_count = len(all_tasks)
            full = build_requests(args.count, args.human_csv, args.lang)
            all_tasks.extend(full[old_count:args.count])
            with plan.open('w', encoding='utf-8-sig', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(all_tasks[0]))
                writer.writeheader()
                writer.writerows(all_tasks)
            print(f'[plan] Extended prompt plan from {old_count} to {args.count} items')
        all_tasks = all_tasks[:args.count]
        for t in all_tasks:
            t['i'] = int(t['i'])
            t['words'] = int(t['words'])
    else:
        all_tasks = build_requests(args.count, args.human_csv, args.lang)
        with plan.open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(all_tasks[0]))
            writer.writeheader()
            writer.writerows(all_tasks)
        print(f'[plan] Saved {len(all_tasks)} distinct prompt specifications to {plan}')
    tasks = [t for t in all_tasks if t['i'] not in existing]
    print(f'[start] model={MODEL}; requested={args.count}; already saved={len(existing)}; left={len(tasks)}')
    if not tasks:
        if len(existing) >= args.count:
            print('[OK] All requested AI samples already on disk.')
        return
    batches = [tasks[i:i + args.batch] for i in range(0, len(tasks), args.batch)]
    semaphore = asyncio.Semaphore(args.parallel)
    lock = asyncio.Lock()
    done = 0
    failures = []
    fields = ['id', 'subject', 'text', 'is_ai', 'source']
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), follow_redirects=False) as client:
        async def job(chunk):
            nonlocal done
            async with semaphore:
                try:
                    rows = await generate_with_fallback(client, chunk, args.retries)
                except RuntimeError as error:
                    # Do not quietly mark failed generations as AI.
                    failures.append((chunk[0]['i'], str(error)))
                    print(f'[ERROR] at id={chunk[0]["i"]}: {error}', flush=True)
                    return
            async with lock:
                is_new = not out.exists()
                with out.open('a', encoding='utf-8-sig' if is_new else 'utf-8', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    if is_new:
                        writer.writeheader()
                    writer.writerows(rows)
                done += len(rows)
                print(f'[saved] {len(existing) + done}/{args.count}', flush=True)
        # Stop early if the very first batch has a fatal configuration issue.
        await job(batches[0])
        if failures:
            raise RuntimeError('First batch failed. Fix API/model settings and rerun; no need to delete the CSV.')
        await asyncio.gather(*(job(chunk) for chunk in batches[1:]))
    if failures:
        raise RuntimeError(f'{len(failures)} AI batches failed; rerun the same command to retry missing IDs.')
    print(f'[done] {out}; completed IDs: {len(previous_ids(out))}/{args.count}')
    print('[!] These are LLM-generated samples, not evidence of detector accuracy.')


def main():
    p = argparse.ArgumentParser(description='Generate diverse LLM-authored email examples via your API')
    p.add_argument('--count', type=int, default=1000)
    p.add_argument('--lang', choices=('en', 'ru'), default='en')
    p.add_argument('--human-csv', default='data/human_emails.csv')
    p.add_argument('--output', default='data/ai_generated.csv')
    p.add_argument('--plan', default='data/ai_prompt_plan_1000.csv')
    p.add_argument('--batch', type=int, default=3)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--retries', type=int, default=2)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == '__main__':
    main()

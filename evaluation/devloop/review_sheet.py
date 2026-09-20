"""Blinded human-review sheet for a set of development-loop runs.

Scenarios are chosen by a hash of their ID (never by any score), arms are
shuffled per question with the same hash, and the arm mapping is written to a
separate file so the reviewer reads answers without knowing which arm wrote them.
"""
import argparse
import hashlib
import json
from pathlib import Path

from arkb.evaluation.v2 import load_dataset

ROOT = Path(__file__).resolve().parents[2]


def hashed(*parts):
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()


def sample(scenarios, slice_name, size, salt):
    chosen = sorted((s for s in scenarios if s['slice'] == slice_name), key=lambda s: hashed(salt, s['id']))
    return chosen[:size]


def v2_reference(case, dataset):
    quotes = []
    for requirement in case['evidence_requirements']:
        for alternative in requirement['alternatives']:
            for evidence_id in alternative:
                span = dataset.evidence[evidence_id]
                quotes.append({'facet': requirement['id'], 'critical': requirement['critical'],
                               'source': span['source'], 'quote': span['quote']})
    return {'task_type': case['task_type'], 'answerability': case['answerability'],
            'required_facts': case.get('required_facts', []), 'expected_behavior': case.get('expected_behavior'),
            'evidence': quotes}


def build(runs, *, devset, salt, v2_size, musique_size):
    scenarios = json.loads((devset / 'scenarios.json').read_text())
    labels = json.loads((devset / 'labels.json').read_text())
    dataset = load_dataset(ROOT / 'evaluation/data/v2/pilot', notes_dir=ROOT / 'evaluation/data/v2/pilot/corpus', allow_provisional=True)
    rows = {arm: {r['scenario']['id']: r for r in (json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines() if line.strip())}
            for arm, path in runs.items()}
    arms = sorted(runs)
    items, mapping = [], {}
    for scenario in sample(scenarios, 'v2', v2_size, salt) + sample(scenarios, 'musique', musique_size, salt):
        sid = scenario['id']
        if scenario['slice'] == 'v2':
            case = next(c for c in dataset.cases if c['id'] == sid)
            reference = v2_reference(case, dataset)
        else:
            gold = labels[sid]['gold']
            reference = {'answerable': gold['answerable'], 'gold_answer': gold['answer'] if gold['answerable'] else None,
                         'answer_aliases': gold['answer_aliases'], 'support_idxs': gold['support_idxs']}
        order = sorted(arms, key=lambda a: hashed(salt, sid, a))
        letters = [chr(ord('A') + i) for i in range(len(order))]
        mapping[sid] = dict(zip(letters, order))
        answers = []
        for letter, arm in zip(letters, order):
            row = rows[arm].get(sid)
            final = ((row or {}).get('result') or {}).get('final') or {}
            answers.append({'letter': letter, 'status': final.get('status', 'missing'), 'answer': final.get('answer'),
                            'cited_sources': sorted({c.get('source') for c in final.get('citations') or [] if isinstance(c, dict)}),
                            'error': (row or {}).get('error')})
        items.append({'id': sid, 'slice': scenario['slice'], 'query': scenario['query'], 'query_zh': scenario.get('query_zh'),
                      'reference': reference, 'answers': answers,
                      'review': {letter: {'correct': None, 'grounded': None, 'notes': ''} for letter in letters}})
    return items, mapping


def markdown(items):
    lines = ['# Blinded review sheet (aw-v1)', '',
             'Answers are labelled A to D in a per-question shuffled order; the mapping to arms is in `arm-mapping.json` and '
             'should be opened only after the sheet is filled. For each answer mark **correct** (yes/partial/no) and '
             '**grounded** (the cited evidence supports the answer: yes/partial/no), then add notes.', '']
    for i, item in enumerate(items, 1):
        lines += [f"## {i}. `{item['id']}` ({item['slice']})", '', f"**Query:** {item['query']}"]
        if item.get('query_zh'):
            lines.append(f"**原文:** {item['query_zh']}")
        ref = item['reference']
        lines.append('')
        if item['slice'] == 'v2':
            lines.append(f"**Task:** {ref['task_type']} / {ref['answerability']}")
            for fact in ref['required_facts']:
                lines.append(f'- required fact: {fact}')
            for q in ref['evidence']:
                lines.append(f"- evidence ({q['facet']}{', critical' if q['critical'] else ''}) `{q['source']}`: {q['quote']}")
        else:
            lines.append(f"**Gold:** answerable={ref['answerable']}; answer={ref['gold_answer']!r}; aliases={ref['answer_aliases']}; support={ref['support_idxs']}")
        for a in item['answers']:
            lines += ['', f"### Answer {a['letter']} (status: {a['status']}; cited: {', '.join(a['cited_sources']) or 'none'})", '']
            text = a['answer'] if a['answer'] is not None else f"(no answer; error: {a['error']})"
            lines += ['> ' + line for line in str(text).splitlines()] or ['> (empty)']
            lines += ['', '- correct: [ ] yes  [ ] partial  [ ] no    grounded: [ ] yes  [ ] partial  [ ] no    notes:']
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, help='ARM=path; repeatable')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--devset', type=Path, default=ROOT / 'evaluation/devloop/devset-v1')
    p.add_argument('--salt', default='aw-v1')
    p.add_argument('--v2', type=int, default=20)
    p.add_argument('--musique', type=int, default=10)
    args = p.parse_args()
    runs = {k: Path(v) for k, v in (item.split('=', 1) for item in args.run)}
    items, mapping = build(runs, devset=args.devset, salt=args.salt, v2_size=args.v2, musique_size=args.musique)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'review-sheet.md').write_text(markdown(items))
    (args.output / 'review-sheet.json').write_text(json.dumps(items, ensure_ascii=False, indent=1) + '\n')
    (args.output / 'arm-mapping.json').write_text(json.dumps({'salt': args.salt, 'runs': {k: str(v) for k, v in runs.items()},
                                                              'mapping': mapping}, ensure_ascii=False, indent=1) + '\n')
    print(f'{len(items)} questions written to {args.output}')


if __name__ == '__main__':
    main()

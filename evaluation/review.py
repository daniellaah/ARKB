"""Blinded human review: export a sheet, import the filled verdicts, report agreement.

Questions are chosen by a hash of their ID (never by any score), the runs'
answers are shuffled per question with the same hash, and the mapping to run
names is written to a separate file so the reviewer reads answers without
knowing which run wrote them. The review itself is done by a person.
"""
import argparse
import json
from pathlib import Path
import re

from .common import ROOT, hash_key, read_json, write_json
from .v2 import load_dataset

DEVSET = ROOT / 'evaluation/devset'
CHOICES = ('yes', 'partial', 'no')


def sample(scenarios, slice_name, size, salt):
    chosen = sorted((s for s in scenarios if s['slice'] == slice_name), key=lambda s: hash_key(salt, s['id']))
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


def build(runs, *, devset=DEVSET, salt, sizes):
    scenarios = read_json(devset / 'scenarios.json')
    labels = read_json(devset / 'labels.json')
    dataset = load_dataset(devset / 'v2-pilot', allow_provisional=True)
    rows = {name: {r['scenario']['id']: r for r in (json.loads(line) for line in (Path(path) / 'results.jsonl').read_text().splitlines() if line.strip())}
            for name, path in runs.items()}
    names = sorted(runs)
    items, mapping = [], {}
    chosen = [s for slice_name, size in sizes.items() for s in sample(scenarios, slice_name, size, salt)]
    for scenario in chosen:
        sid = scenario['id']
        if scenario['slice'] == 'v2':
            reference = v2_reference(dataset.case(sid), dataset)
        elif scenario['slice'] == 'musique':
            gold = labels[sid]['gold']
            reference = {'answerable': gold['answerable'], 'gold_answer': gold['answer'] if gold['answerable'] else None,
                         'answer_aliases': gold['answer_aliases'], 'support_idxs': gold['support_idxs']}
        else:
            reference = {k: v for k, v in labels[sid].items() if k != 'source_map'}
        order = sorted(names, key=lambda a: hash_key(salt, sid, a))
        letters = [chr(ord('A') + i) for i in range(len(order))]
        mapping[sid] = dict(zip(letters, order))
        answers = []
        for letter, name in zip(letters, order):
            row = rows[name].get(sid)
            final = ((row or {}).get('result') or {}).get('final') or {}
            answers.append({'letter': letter, 'status': final.get('status', 'missing'), 'answer': final.get('answer'),
                            'cited_sources': sorted({c.get('source') for c in final.get('citations') or [] if isinstance(c, dict)}),
                            'error': (row or {}).get('error')})
        items.append({'id': sid, 'slice': scenario['slice'], 'query': scenario['query'], 'query_zh': scenario.get('query_zh'),
                      'reference': reference, 'answers': answers})
    return items, mapping


def markdown(items):
    lines = ['# Blinded review sheet', '',
             'Answers are labelled A, B, ... in a per-question shuffled order; the mapping to runs is in `mapping.json` and '
             'should be opened only after the sheet is filled. For each answer mark **correct** (yes/partial/no) and '
             '**grounded** (the cited evidence supports the answer: yes/partial/no) by putting an x in one box each, '
             'then add notes on the same line.', '']
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
        elif item['slice'] == 'musique':
            lines.append(f"**Gold:** answerable={ref['answerable']}; answer={ref['gold_answer']!r}; aliases={ref['answer_aliases']}; support={ref['support_idxs']}")
        else:
            lines.append(f'**Labels:** {json.dumps(ref, ensure_ascii=False)}')
        for a in item['answers']:
            lines += ['', f"### Answer {a['letter']} (status: {a['status']}; cited: {', '.join(a['cited_sources']) or 'none'})", '']
            text = a['answer'] if a['answer'] is not None else f"(no answer; error: {a['error']})"
            lines += ['> ' + line for line in str(text).splitlines()] or ['> (empty)']
            lines += ['', '- correct: [ ] yes  [ ] partial  [ ] no    grounded: [ ] yes  [ ] partial  [ ] no    notes:']
        lines.append('')
    return '\n'.join(lines) + '\n'


_ANSWER = re.compile(r'^### Answer ([A-Z]) ')
_QUESTION = re.compile(r'^## \d+\. `([^`]+)`')
_VERDICT = re.compile(r'^- correct: (.*?)\s{2,}grounded: (.*?)\s{2,}notes:(.*)$')


def _choice(field):
    marked = [name for name in CHOICES if re.search(r'\[[xX✓]\]\s*' + name, field)]
    if len(marked) > 1:
        raise ValueError('More than one box marked: ' + field)
    return marked[0] if marked else None


def parse_sheet(text):
    """Verdicts from a filled sheet: {question_id: {letter: {correct, grounded, notes}}}; unmarked answers are absent."""
    verdicts, question, letter = {}, None, None
    for line in text.splitlines():
        if m := _QUESTION.match(line):
            question, letter = m.group(1), None
        elif m := _ANSWER.match(line):
            letter = m.group(1)
        elif (m := _VERDICT.match(line)) and question and letter:
            correct, grounded, notes = _choice(m.group(1)), _choice(m.group(2)), m.group(3).strip()
            if correct or grounded or notes:
                verdicts.setdefault(question, {})[letter] = {'correct': correct, 'grounded': grounded, 'notes': notes}
    return verdicts


def import_verdicts(sheet, mapping_path, destination, *, reviewer):
    """Unblind a filled sheet into per-run verdicts; a question with no marks is left out."""
    mapping = read_json(mapping_path)
    verdicts = parse_sheet(Path(sheet).read_text())
    by_run = {}
    for question, answers in verdicts.items():
        for letter, verdict in answers.items():
            run = mapping['mapping'][question][letter]
            by_run.setdefault(run, {})[question] = verdict
    record = {'reviewer': reviewer, 'sheet': str(sheet), 'salt': mapping['salt'], 'runs': mapping['runs'],
              'questions_reviewed': sorted(verdicts), 'verdicts': by_run}
    write_json(destination, record)
    return record


def agreement(record, run_name, results_path):
    """How often the run's automatic outcome (answered/partial against the rest) matched the human `correct` verdict."""
    rows = {r['scenario']['id']: r for r in (json.loads(line) for line in Path(results_path).read_text().splitlines() if line.strip())}
    cells = []
    for question, verdict in record['verdicts'].get(run_name, {}).items():
        if verdict['correct'] is None or question not in rows:
            continue
        status = ((rows[question].get('result') or {}).get('final') or {}).get('status', 'error')
        cells.append({'id': question, 'status': status, 'correct': verdict['correct'], 'grounded': verdict['grounded'],
                      'agree': (status in ('answered', 'partial')) == (verdict['correct'] in ('yes', 'partial'))})
    return {'run': run_name, 'reviewed': len(cells), 'agreement': (sum(c['agree'] for c in cells) / len(cells)) if cells else None,
            'correct': {k: sum(c['correct'] == k for c in cells) for k in CHOICES},
            'grounded': {k: sum(c['grounded'] == k for c in cells) for k in CHOICES}, 'cells': cells}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export', help='write a blinded sheet for several runs')
    e.add_argument('--run', action='append', required=True, help='NAME=path; repeatable')
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--devset', type=Path, default=DEVSET)
    e.add_argument('--salt', required=True, help='review name; fixes the sample and the shuffle')
    e.add_argument('--size', action='append', default=None, help='SLICE=count; repeatable; default v2=20, musique=10')
    i = sub.add_parser('import', help='unblind a filled sheet into per-run verdicts')
    i.add_argument('sheet', type=Path)
    i.add_argument('--mapping', type=Path, required=True)
    i.add_argument('--output', type=Path, required=True)
    i.add_argument('--reviewer', required=True)
    a = sub.add_parser('agreement', help='automatic outcome against human verdicts for one run')
    a.add_argument('verdicts', type=Path)
    a.add_argument('--run', required=True, help='run name as used at export')
    a.add_argument('--results', type=Path, required=True, help='that run\'s results.jsonl')
    args = p.parse_args()
    if args.command == 'export':
        runs = {k: Path(v) for k, v in (item.split('=', 1) for item in args.run)}
        sizes = dict((k, int(v)) for k, v in (item.split('=', 1) for item in args.size)) if args.size else {'v2': 20, 'musique': 10}
        items, mapping = build(runs, devset=args.devset, salt=args.salt, sizes=sizes)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'sheet.md').write_text(markdown(items))
        write_json(args.output / 'sheet.json', items)
        write_json(args.output / 'mapping.json', {'salt': args.salt, 'runs': {k: str(v) for k, v in runs.items()}, 'mapping': mapping})
        print(f'{len(items)} questions written to {args.output}')
    elif args.command == 'import':
        record = import_verdicts(args.sheet, args.mapping, args.output, reviewer=args.reviewer)
        print(json.dumps({'questions_reviewed': len(record['questions_reviewed']), 'runs': sorted(record['verdicts'])}))
    else:
        print(json.dumps(agreement(read_json(args.verdicts), args.run, args.results), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Blinded review sheet: hash-chosen sample, per-question shuffled runs, mapping kept apart, verdict import."""
import json

import pytest

from evaluation.common import hash_key, write_json
from evaluation.review import agreement, import_verdicts, markdown, parse_sheet, sample


def test_sample_is_by_id_hash_only_and_order_independent():
    scenarios = [{'id': f'q{i}', 'slice': 'v2' if i % 2 else 'musique'} for i in range(40)]
    chosen = sample(scenarios, 'v2', 5, 'salt')
    assert len(chosen) == 5 and all(s['slice'] == 'v2' for s in chosen)
    assert [s['id'] for s in chosen] == [s['id'] for s in sample(list(reversed(scenarios)), 'v2', 5, 'salt')]
    assert [s['id'] for s in chosen] != [s['id'] for s in sample(scenarios, 'v2', 5, 'other-salt')]
    assert hash_key('a', 'b') != hash_key('a', 'c')


def test_markdown_hides_arms_and_quotes_answers():
    items = [{'id': 'q1', 'slice': 'musique', 'query': 'Who?', 'query_zh': None,
              'reference': {'answerable': True, 'gold_answer': 'X', 'answer_aliases': [], 'support_idxs': [1]},
              'answers': [{'letter': 'A', 'status': 'answered', 'answer': 'X\nbecause', 'cited_sources': ['a.md'], 'error': None},
                          {'letter': 'B', 'status': 'error', 'answer': None, 'cited_sources': [], 'error': {'type': 'E'}}]}]
    text = markdown(items)
    assert '### Answer A (status: answered; cited: a.md)' in text and '> X\n> because' in text
    assert '(no answer; error:' in text and 'A-All' not in text and 'F-H' not in text


def test_filled_sheet_is_unblinded_into_per_run_verdicts_and_agreement(tmp_path):
    filled = '''# Blinded review sheet

## 1. `q1` (musique)

### Answer A (status: answered; cited: a.md)

> X

- correct: [x] yes  [ ] partial  [ ] no    grounded: [ ] yes  [X] partial  [ ] no    notes: fine

### Answer B (status: insufficient_evidence; cited: none)

> nothing

- correct: [ ] yes  [ ] partial  [x] no    grounded: [ ] yes  [ ] partial  [ ] no    notes:

## 2. `q2` (v2)

### Answer A (status: answered; cited: none)

> untouched

- correct: [ ] yes  [ ] partial  [ ] no    grounded: [ ] yes  [ ] partial  [ ] no    notes:
'''
    verdicts = parse_sheet(filled)
    assert verdicts == {'q1': {'A': {'correct': 'yes', 'grounded': 'partial', 'notes': 'fine'},
                               'B': {'correct': 'no', 'grounded': None, 'notes': ''}}}
    (tmp_path / 'sheet.md').write_text(filled)
    write_json(tmp_path / 'mapping.json', {'salt': 's', 'runs': {'agent': 'x', 'fixed': 'y'}, 'mapping': {'q1': {'A': 'agent', 'B': 'fixed'}, 'q2': {'A': 'fixed'}}})
    record = import_verdicts(tmp_path / 'sheet.md', tmp_path / 'mapping.json', tmp_path / 'verdicts.json', reviewer='me')
    assert record['verdicts'] == {'agent': {'q1': {'correct': 'yes', 'grounded': 'partial', 'notes': 'fine'}},
                                  'fixed': {'q1': {'correct': 'no', 'grounded': None, 'notes': ''}}}
    rows = [{'scenario': {'id': 'q1'}, 'result': {'final': {'status': 'answered'}}}, {'scenario': {'id': 'q2'}, 'result': {'final': {'status': 'answered'}}}]
    (tmp_path / 'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    assert agreement(record, 'agent', tmp_path / 'results.jsonl')['agreement'] == 1.0
    assert agreement(record, 'fixed', tmp_path / 'results.jsonl')['agreement'] == 0.0
    with pytest.raises(ValueError, match='More than one'):
        parse_sheet(filled.replace('[x] yes  [ ] partial  [ ] no    grounded', '[x] yes  [x] partial  [ ] no    grounded'))

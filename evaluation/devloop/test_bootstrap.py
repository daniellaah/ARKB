"""Generic paired bootstrap checks; synthetic scores, no services."""
import json

import numpy as np
import pytest

from arkb.evaluation.external import write_json
from evaluation.devloop.bootstrap import analyze, markdown, metric_value, paired_bootstrap, units_for


def test_paired_bootstrap_is_deterministic_stratified_and_counts_wins():
    rng = np.random.default_rng(1)
    units = [{'id': f'q{i}', 'stratum': 'a' if i < 30 else 'b',
              'values': {'X': float(rng.normal(0.5, 0.1)), 'Y': float(rng.normal(0.6, 0.1))}} for i in range(60)]
    first = paired_bootstrap(units, [('Y', 'X'), ('X', 'Y')], resamples=2000)
    second = paired_bootstrap(list(reversed(units)), [('Y', 'X'), ('X', 'Y')], resamples=2000)
    assert first == second and first['stratum_sizes'] == {'a': 30, 'b': 30}
    gain, loss = first['contrasts']
    assert gain['comparison'] == 'Y minus X' and gain['mean_difference'] == pytest.approx(-loss['mean_difference'])
    lo, hi = gain['nominal_95_ci']
    assert lo < gain['mean_difference'] < hi and 0 < lo
    assert gain['wins'] + gain['ties'] + gain['losses'] == 60 and gain['wins'] == loss['losses']
    exact = paired_bootstrap([{'id': 'q', 'stratum': 's', 'values': {'X': 1.0, 'Y': 1.0}}], [('Y', 'X')], resamples=10)
    assert exact['contrasts'][0]['ties'] == 1 and exact['contrasts'][0]['nominal_95_ci'] == [0.0, 0.0]
    with pytest.raises(ValueError, match='twice'):
        paired_bootstrap(units + [units[0]], [('Y', 'X')], resamples=10)
    with pytest.raises(ValueError, match='without a run'):
        paired_bootstrap(units, [('Z', 'X')], resamples=10)


def scores(value, *, elapsed=1000, answered=True, tokens=7):
    return {'final_status': 'answered', 'tool_names': ['search', 'read'], 'positive_recall_delivered': value,
            'behavior': {'answered': answered, 'abstained': None},
            'costs': {'elapsed_ms': elapsed, 'model_requests': 3, 'usage': {'prompt_eval_count': {'known_total': tokens}, 'eval_count': {'known_total': None}}}}


def test_metric_paths_and_missing_values_drop_the_pair():
    s = scores(0.5)
    assert metric_value(s, 'elapsed_s') == 1.0 and metric_value(s, 'tool_calls') == 2.0
    assert metric_value(s, 'costs.model_requests') == 3.0 and metric_value(s, 'prompt_tokens') == 7.0
    assert metric_value(s, 'eval_tokens') is None and metric_value(s, 'behavior.abstained') is None
    assert metric_value(s, 'behavior.answered') == 1.0 and metric_value(s, 'missing.path') is None
    rows = {'A': {'q1': {'scenario': {'id': 'q1', 'slice': 'recall-fiqa'}, 'scores': scores(0.5)},
                  'q2': {'scenario': {'id': 'q2', 'slice': 'recall-fiqa'}, 'scores': scores(None)}},
            'B': {'q1': {'scenario': {'id': 'q1', 'slice': 'recall-fiqa'}, 'scores': scores(0.25)},
                  'q2': {'scenario': {'id': 'q2', 'slice': 'recall-fiqa'}, 'scores': scores(0.75)},
                  'q3': {'scenario': {'id': 'q3', 'slice': 'recall-fiqa'}, 'scores': scores(0.75)}}}
    units = units_for(rows, 'positive_recall_delivered')
    assert [u['id'] for u in units] == ['q1'] and units[0]['values'] == {'A': 0.5, 'B': 0.25}
    assert len(units_for(rows, 'elapsed_s')) == 2 and units_for(rows, 'elapsed_s', slices=['v2']) == []


def test_analyze_writes_per_slice_and_pooled_cost_blocks(tmp_path):
    for arm, value, elapsed in (('A-All', 0.6, 3000), ('F-H', 0.4, 1000)):
        d = tmp_path / arm
        d.mkdir()
        rows = []
        for name in ('recall-fiqa', 'recall-nfcorpus'):
            for i in range(4):
                rows.append({'scenario': {'id': f'{name}-{i}', 'slice': name}, 'scores': scores(value + i / 100, elapsed=elapsed + i)})
        (d / 'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
        write_json(d / 'run.json', {'label': arm, 'model': 'm', 'think': True, 'git_head': 'abcdef0123', 'dirty': False})
    report = analyze({'A-All': tmp_path / 'A-All', 'F-H': tmp_path / 'F-H'}, [('A-All', 'F-H')], resamples=200)
    keys = {(b['slice'], b['metric']) for b in report['blocks']}
    assert ('recall-fiqa', 'positive_recall_delivered') in keys and ('all (stratified)', 'elapsed_s') in keys
    pooled = next(b for b in report['blocks'] if b['slice'] == 'all (stratified)' and b['metric'] == 'elapsed_s')
    assert pooled['n'] == 8 and pooled['stratum_sizes'] == {'recall-fiqa': 4, 'recall-nfcorpus': 4}
    assert pooled['contrasts'][0]['mean_difference'] == pytest.approx(2.0) and pooled['contrasts'][0]['wins'] == 8
    text = markdown(report)
    assert '| recall-fiqa | positive_recall_delivered | 4 |' in text and 'A-All minus F-H' in text

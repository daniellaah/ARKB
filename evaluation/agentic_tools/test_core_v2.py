"""Core v2 design, pause rules, competitor detection and repeat-subset statistics; no labels."""
from copy import deepcopy
import json

import pytest

from arkb.evaluation.external import digest, write_json
from .contract import ARMS
from .core_design import (COUNTS, PRIMARY_TOTAL, REPEAT_SUBSET, TOTAL, core_schedule, precision, repeat_ids,
                          required_questions, selection_for_core)
from .readiness import POLICY
from . import runner
from .runner import CoreGuard, gpu_competitors
from .selection import build_selection, attempt_key, key
from .statistics import query_units, session_variability
from .analyze import registered_analysis
from .core_v2 import build_scenarios, check_tests


def original_selection():
    populations = {'browsecomp-plus': [str(i) for i in range(1, 831)], 'fiqa': [f'f{i}' for i in range(648)],
                   'nfcorpus': [f'n{i}' for i in range(323)]}
    pairs = [f'{hop}hop__{i}' for hop, count in (('2', 34), ('3', 33), ('4', 33)) for i in range(count)]
    return build_selection(populations, pairs)


def test_selection_excludes_pilot_and_exposed_questions_by_id_rule():
    original = original_selection()
    first_core = original['tracks']['browsecomp-plus'][0]['core'][0]
    selected = selection_for_core(original, {'browsecomp-plus': [first_core]})
    for dataset, strata in selected['tracks'].items():
        for pilot_stratum, stratum in zip(original['tracks'][dataset], strata, strict=True):
            assert stratum['pilot'] == []
            assert not set(pilot_stratum['pilot']) & set(stratum['core'])
            assert len(stratum['core']) == (10 if dataset == 'musique' else COUNTS[dataset])
            remaining = [x for x in pilot_stratum['pilot'] + pilot_stratum['core'] + pilot_stratum['reserves']
                         if x not in pilot_stratum['pilot'] and x != first_core]
            assert stratum['core'] + stratum['reserves'] == sorted(remaining, key=lambda x: (key(dataset, stratum['stratum'], x), x))
    browse = selected['tracks']['browsecomp-plus'][0]
    assert first_core not in browse['core'] and first_core in browse['development_excluded']
    assert browse['eligible_count'] == 825
    with pytest.raises(ValueError, match='Insufficient'):
        selection_for_core(original, {'fiqa': original['tracks']['fiqa'][0]['reserves'] + original['tracks']['fiqa'][0]['core']})


def test_schedule_shape_rotation_pairing_and_repeat_subset():
    selection = selection_for_core(original_selection(), {})
    schedule = core_schedule(selection)
    assert len(schedule) == TOTAL and sum(r['repetition'] == 0 for r in schedule) == PRIMARY_TOTAL
    assert len({attempt_key('p', r) for r in schedule}) == TOTAL
    repeated = set(repeat_ids(selection))
    assert repeated == set(selection['tracks']['browsecomp-plus'][0]['core'][:REPEAT_SUBSET['questions']])
    for row in schedule:
        expected = set(REPEAT_SUBSET['repetitions']) if row['dataset'] == 'browsecomp-plus' and row['id'] in repeated else {0}
        assert row['repetition'] in expected
    assert [r['arm'] for r in schedule[:7]] == [a.id for a in ARMS]
    assert schedule[7]['repetition'] == 1 and schedule[7]['arm'] == ARMS[1].id
    unit_one = [r for r in schedule if r['unit_index'] == 1]
    assert unit_one[0]['arm'] == ARMS[1].id
    mu = [r for r in schedule if r['dataset'] == 'musique']
    assert len(mu) == 420
    for a, b in zip(mu[::2], mu[1::2], strict=True):
        assert a['arm'] == b['arm'] and (a['variant'], b['variant']) == ('v0', 'v1')


def test_precision_targets_do_not_depend_on_answers():
    assert required_questions(.10, .5) == 312
    for rho in (0, .5, 1):
        assert precision(320, 1, .5, rho)['effective_n'] == 320
        assert precision(100, 3, .5, rho)['effective_n'] == pytest.approx(100 / (rho + (1 - rho) / 3))
    assert precision(320, 1, .5, 0)['normal_half_width'] < .10
    assert not precision(320, 1, .25, 0)['normal_power_at_assumed_delta'] > .5
    with pytest.raises(ValueError):
        precision(0, 1, .5, 0)
    with pytest.raises(ValueError):
        precision(10, 1, .01, 0, delta=.05)


CORE_POLICY = {'max_operational_failures_per_attempt': 3, 'window_min_valid_calls': 200,
               'window_max_operational_rate': .01, 'trajectory_time_cap_hours': 1}


def record(key, calls, *, failed=0, elapsed_ms=1000, error=None, competing=None):
    tools = [{'name': 'search', 'index': i, 'executed': True, 'status': 'recoverable_error' if i < failed else 'success',
              'elapsed_ms': 31000 if i < failed else 100, 'arguments': {'mode': 'bm25'},
              'error': {'code': 'search_timeout'} if i < failed else None} for i in range(calls)]
    return {'key': key, 'schedule': {'dataset': 'browsecomp-plus', 'arm': 'A-B'}, 'elapsed_ms': elapsed_ms,
            'error': error, 'competing_processes_after': competing or [],
            'result': {'final': {'status': 'insufficient_evidence'}, 'observation': {'models': [], 'tools': tools}}}


def test_single_failures_are_outcomes_but_patterns_pause():
    guard = CoreGuard(POLICY, CORE_POLICY)
    assert guard.observe(record('a', 5, failed=1)) == []
    assert guard.observe(record('b', 5, failed=2)) == []
    assert guard.observe(record('c', 5, failed=3)) == ['repeated_operational_failures:c']
    state = guard.state()
    assert state['window_valid_calls'] == 15 and state['window_operational_failures'] == 6


def test_rate_rule_needs_minimum_calls_then_one_percent():
    guard = CoreGuard(POLICY, CORE_POLICY)
    for i in range(39):
        assert guard.observe(record(str(i), 5)) == []
    assert guard.observe(record('x', 5, failed=2)) == []  # 2/200 == 1 percent, not above
    assert guard.observe(record('y', 5, failed=1)) == ['operational_failure_rate:3/205']


def test_reviewed_attempts_leave_the_window_and_execution_failures_pause():
    guard = CoreGuard(POLICY, CORE_POLICY, reviewed=['c'])
    assert guard.observe(record('c', 5, failed=3)) == []
    assert guard.state()['reviewed_attempts'] == 1 and guard.state()['window_valid_calls'] == 0
    assert guard.observe(record('d', 2, error={'type': 'RuntimeError'})) == ['unobserved_execution_failure:d']
    assert guard.observe(record('e', 2, competing=['1 mlx_lm.lora'])) == ['gpu_competition_during_attempt:e']


def test_time_cap_counts_measured_trajectory_seconds_and_explicit_extension():
    guard = CoreGuard(POLICY, CORE_POLICY)
    assert guard.observe(record('a', 1, elapsed_ms=3599_000)) == []
    assert guard.observe(record('b', 1, elapsed_ms=1_000)) == ['trajectory_time_cap_reached']
    extended = CoreGuard(POLICY, CORE_POLICY, cap_hours=2)
    assert extended.observe(record('a', 1, elapsed_ms=3600_000)) == []
    assert extended.state()['time_cap_hours'] == 2


def test_quality_does_not_change_guard_decisions():
    guard_a, guard_b = CoreGuard(POLICY, CORE_POLICY), CoreGuard(POLICY, CORE_POLICY)
    a = record('a', 5, failed=3)
    b = deepcopy(a)
    b['result']['final'] = {'status': 'answered', 'answer': 'arbitrary'}
    assert guard_a.observe(a) == guard_b.observe(b)


def ps_listing(rows):
    return '\n'.join(f'{pid} Thu Sep 17 22:00:{i:02d} 2026 {command}' for i, (pid, command) in enumerate(rows)) + '\n'


def test_foreign_python_with_metal_device_is_a_competitor(monkeypatch):
    rows = [(10, '/Users/x/GitHub/transformers/.venv/bin/python train.py'),
            (11, '/Users/x/GitHub/agentic-retrieval-for-knowledge-bases/.venv/bin/python -m evaluation.agentic_tools.runner'),
            (12, '/bin/zsh -c uv run mlx_lm.lora --config c.yaml'),
            (13, 'uv run mlx_lm.lora --config c.yaml'),
            (14, '/opt/homebrew/bin/python3 idle.py')]
    checked = []
    def metal(pid):
        checked.append(pid)
        return pid == 10
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: ps_listing(rows))
    runner._METAL_CACHE.clear()
    found = gpu_competitors(now=0, metal_check=metal)
    assert found == ['10 /Users/x/GitHub/transformers/.venv/bin/python train.py [metal-device]', '13 uv run mlx_lm.lora --config c.yaml']
    assert sorted(checked) == [10, 14]
    checked.clear()
    assert gpu_competitors(now=100, metal_check=metal)[0].startswith('10 ')
    assert checked == []  # cached
    gpu_competitors(now=400, metal_check=metal)
    assert checked == [14]  # negative rechecked after the interval; positive kept
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: ps_listing(rows[1:]))
    assert gpu_competitors(now=401, metal_check=metal) == ['13 uv run mlx_lm.lora --config c.yaml']
    assert (10, 'Thu Sep 17 22:00:00 2026') not in runner._METAL_CACHE


def rows_for(question_values, repetitions):
    result = []
    for q, per_arm in question_values.items():
        for arm in ARMS:
            for rep in repetitions:
                value = per_arm[arm.id]
                result.append({'schedule': {'id': q, 'arm': arm.id, 'stratum': 'all', 'variant': 'v0', 'repetition': rep},
                               'score': value[rep] if isinstance(value, (list, tuple)) else value})
    return result


def test_single_repetition_units_and_repeat_subset_decomposition():
    rows = rows_for({'q1': {a.id: 1.0 for a in ARMS}, 'q2': {a.id: 0.0 for a in ARMS}}, (0,))
    units = query_units(rows, 'score', repetitions=(0,))
    assert len(units) == 2 and units[0]['values']['A-All'] == 1.0
    with pytest.raises(ValueError, match='missing a repetition'):
        query_units(rows, 'score')
    stable = rows_for({'q1': {a.id: 1.0 for a in ARMS}, 'q2': {a.id: 0.0 for a in ARMS}}, (0, 1, 2))
    result = session_variability(stable, 'score')
    assert result['status'] == 'complete' and result['arms']['A-All']['repeat_correlation'] == 1.0
    assert result['arms']['A-All']['questions_with_identical_repetitions'] == 2
    noisy = rows_for({'q1': {a.id: (1.0, 0.0, 1.0) for a in ARMS}, 'q2': {a.id: (0.0, 1.0, 0.0) for a in ARMS}}, (0, 1, 2))
    noisy_result = session_variability(noisy, 'score')
    assert noisy_result['arms']['A-All']['repeat_correlation'] == 0.0
    assert noisy_result['primary_contrasts']['A-All minus A-M']['repeat_correlation'] is None
    pending = deepcopy(stable)
    pending[0]['score'] = None
    assert session_variability(pending, 'score')['status'] == 'pending_scores'
    with pytest.raises(ValueError, match='incomplete'):
        session_variability(stable[:-1], 'score')


def test_registered_analysis_roles_come_from_protocol():
    assert registered_analysis({}) == ((0, 1, 2), None)
    primary, subset = registered_analysis({'statistics': {'primary_repetitions': [0], 'repeat_subset': {
        'dataset': 'browsecomp-plus', 'ids': ['1', '2'], 'repetitions': [0, 1, 2]}}})
    assert primary == (0,) and subset == {'dataset': 'browsecomp-plus', 'ids': {'1', '2'}, 'repetitions': (0, 1, 2)}


def test_fresh_scenarios_come_only_from_pinned_queries_and_never_from_pilot(tmp_path):
    import hashlib
    data = tmp_path / 'data'
    data.mkdir()
    queries = {'1': 'first question', '2': 'second question', '3': 'third question'}
    write_json(data / 'queries.json', queries)
    write_json(data / 'manifest.json', {'query_ids': list(queries), 'files': {'queries.json': digest(data / 'queries.json')}})
    inputs = {'browsecomp-plus': {'data': str(data), 'data_manifest_sha256': digest(data / 'manifest.json')}}
    pilot = tmp_path / 'pilot'
    (pilot / 'inference').mkdir(parents=True)
    def sid(identity):
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    existing = {sid(('browsecomp-plus', '1', 'v0')): {'scenario_id': sid(('browsecomp-plus', '1', 'v0')), 'dataset': 'browsecomp-plus',
                                                     'id': '1', 'variant': 'v0', 'split': 'pilot', 'query': 'first question'},
                sid(('browsecomp-plus', '2', 'v0')): {'scenario_id': sid(('browsecomp-plus', '2', 'v0')), 'dataset': 'browsecomp-plus',
                                                     'id': '2', 'variant': 'v0', 'split': 'core', 'query': 'second question'}}
    write_json(pilot / 'inference/scenarios.json', existing)
    schedule = [{'dataset': 'browsecomp-plus', 'id': i, 'variant': 'v0'} for i in ('2', '3')]
    scenarios = build_scenarios(pilot, schedule, inputs)
    assert len(scenarios) == 2 and scenarios[sid(('browsecomp-plus', '3', 'v0'))]['query'] == 'third question'
    with pytest.raises(ValueError, match='pilot'):
        build_scenarios(pilot, [{'dataset': 'browsecomp-plus', 'id': '1', 'variant': 'v0'}], inputs)
    with pytest.raises(ValueError, match='Missing prepared scenario'):
        build_scenarios(pilot, [{'dataset': 'fiqa', 'id': 'f1', 'variant': 'v0'}], inputs)


def test_registration_requires_clean_regression_run(tmp_path):
    good = tmp_path / 'good.xml'
    good.write_text('<testsuites><testsuite tests="1300" failures="0" errors="0" skipped="0"/></testsuites>')
    assert check_tests(good) == 1300
    bad = tmp_path / 'bad.xml'
    bad.write_text('<testsuites><testsuite tests="1300" failures="1" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(ValueError):
        check_tests(bad)

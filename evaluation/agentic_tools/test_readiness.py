"""Operational gates must see failures even when final answers are schema-valid."""
from copy import deepcopy
import signal
from types import SimpleNamespace
from dataclasses import dataclass
import json

import pytest

from arkb.evaluation.external import write_json, digest
from .readiness import POLICY, assess, operational_failures, execution_failure
from .stopping import StopController


def record(*, status='success', code=None, seconds=1, name='match', args=None):
    return {'key': 'attempt', 'schedule': {'dataset': 'browsecomp-plus', 'arm': 'A-All'},
            'result': {'final': {'status': 'insufficient_evidence'}, 'observation': {
                'models': [], 'tools': [{'name': name, 'index': 0, 'executed': True, 'status': status,
                'elapsed_ms': seconds * 1000 if seconds is not None else None,
                'arguments': args or {}, 'error': {'code': code} if code else None}]}}}


def test_recoverable_timeout_blocks_even_with_valid_final():
    row = record(status='recoverable_error', code='exact_timeout', seconds=30)
    assert operational_failures(row, POLICY)
    assert assess([row], POLICY, complete=True)['status'] == 'failed'


@pytest.mark.parametrize('code', ['invalid_arguments', 'invalid_pattern', 'invalid_reference', 'unknown_tool'])
def test_model_input_mistakes_are_retained_and_separate(code):
    row = record(status='recoverable_error', code=code)
    assert not operational_failures(row, POLICY)
    gate = assess([record(), row], POLICY, complete=True)
    assert gate['status'] == 'passed'
    assert gate['groups'][0]['calls'] == 2
    assert gate['groups'][0]['input_errors'] == 1


def test_missing_known_filename_is_distinct_from_missing_bound_reference():
    guessed = record(status='recoverable_error', code='source_unavailable', name='read', args={'source': 'missing.md'})
    bound = record(status='recoverable_error', code='source_unavailable', name='read', args={'ref': 'ev_1'})
    assert not operational_failures(guessed, POLICY)
    assert operational_failures(bound, POLICY)


@pytest.mark.parametrize('seconds', [None, float('nan'), float('inf'), -1, 30])
def test_missing_invalid_or_late_timing_cannot_pass(seconds):
    assert operational_failures(record(seconds=seconds), POLICY)


def test_slow_tool_cannot_hide_among_fast_calls_from_another_tool():
    rows = [record(seconds=.01, name='read') for _ in range(100)] + [record(seconds=16)]
    assert not operational_failures(rows[-1], POLICY)  # final p95 gate, not an immediate timeout
    assert assess(rows, POLICY, complete=True)['status'] == 'failed'


def test_partial_or_different_answer_quality_does_not_manipulate_gate():
    a = record()
    b = deepcopy(a)
    b['result']['final'] = {'status': 'answered', 'answer': 'arbitrary'}
    assert assess([a], POLICY, complete=True) == assess([b], POLICY, complete=True)
    assert assess([a], POLICY, complete=False)['status'] == 'pending'


def test_provider_failures_block_but_invalid_model_final_is_a_model_outcome():
    row = record()
    row['result']['observation']['models'] = [{'status': 'fatal_error', 'error': {'stage': 'model_request'}}]
    assert execution_failure(row)
    row['result']['observation']['models'][0]['error']['stage'] = 'model_protocol'
    assert not execution_failure(row)


def test_signal_stop_finishes_work_and_persists_across_restart(tmp_path):
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    path = tmp_path / 'stop-request.json'
    with StopController(path) as stop:
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGINT)
        assert stop.requested()
        assert stop.persist()['reason'] == 'SIGTERM'
    assert {s: signal.getsignal(s) for s in before} == before
    with StopController(path) as restarted:
        assert restarted.requested()
    assert json.loads(path.read_text())['reason'] == 'SIGTERM'


def test_existing_stop_file_never_overwritten(tmp_path):
    path = tmp_path / 'stop-request.json'
    path.write_text('user supplied stop')
    with StopController(path) as stop:
        stop.request('automatic gate')
        stop.persist()
    assert path.read_text() == 'user supplied stop'


def test_registration_waits_for_training_without_provider_calls(tmp_path, monkeypatch):
    from . import pilot_pipeline, preflight
    parent, out = tmp_path / 'parent', tmp_path / 'out'
    parent.mkdir()
    out.mkdir()
    write_json(parent / 'protocol.json', {})
    monkeypatch.setattr(pilot_pipeline, 'gpu_competitors', lambda: ['training process'])
    monkeypatch.setattr(preflight, 'main', lambda *a: pytest.fail('Provider must not run during training'))
    result = pilot_pipeline.register_if_available(out, parent, tmp_path)
    assert result['status'] == 'waiting_for_gpu'
    assert not (out / 'protocol.json').exists()


def test_registration_does_not_take_another_inference_owners_lock(tmp_path):
    import fcntl
    from .pilot_pipeline import register_if_available
    write_json(tmp_path / 'protocol.json', {})
    with (tmp_path / 'inference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert register_if_available(tmp_path, tmp_path, tmp_path)['status'] == 'waiting_for_inference_lock'


def test_runner_pause_commits_one_attempt_and_resume_never_repeats_it(tmp_path, monkeypatch):
    from . import runner
    from .selection import attempt_key
    schedule = [{'dataset': 'fixture', 'id': str(i), 'variant': 'v0', 'arm': 'F-S', 'repetition': 0} for i in range(2)]
    protocol = {'schema': 'agentic-tools-v1-executable-pilot', 'phase': 'pilot', 'pilot_attempts': 2,
                'files': {}, 'tokenizer_cache': str(tmp_path), 'qdrant_url': 'http://localhost:6340',
                'models': {'chat': {}, 'embedding': {}}, 'measured_source': str(tmp_path)}
    @dataclass
    class Manifest:
        index_version: str = 'fixture-index'
    manifest = Manifest()
    entry = {'backend': {'input': {'tokenizer': 'token-hash'}}, 'index_manifest': {'index_version': 'fixture-index'},
             'data': str(tmp_path), 'corpus': str(tmp_path), 'sqlite': str(tmp_path / 'index.sqlite'),
             'sqlite_sha256': digest(__file__), 'vault_id': 'fixture'}
    (tmp_path / 'index.sqlite').write_bytes(__import__('pathlib').Path(__file__).read_bytes())
    for name, value in {'protocol.json': protocol, 'pilot-schedule.json': schedule, 'inputs.json': {'fixture': entry, 'browsecomp-plus': entry},
                        'context-indexes.json': {}, 'measured-source-manifest.json': {}}.items():
        write_json(tmp_path / name, value)
    (tmp_path / 'inference').mkdir()
    write_json(tmp_path / 'inference/scenarios.json', {str(i): {'dataset': 'fixture', 'id': str(i), 'variant': 'v0',
               'scenario_id': str(i), 'query': 'q'} for i in range(2)})
    class Resource:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def tokenizer(self): return SimpleNamespace(encode=lambda *a, **k: SimpleNamespace(ids=[]))
        def retrieval_engine(self, *a, **k): return object()
        def agent_tools(self, **k): return SimpleNamespace(_exact=SimpleNamespace(close=lambda: None))
        def active_manifest(self, *a): return manifest
    class Client:
        def __init__(self, *a): self.http = SimpleNamespace(get=lambda *a: SimpleNamespace(raise_for_status=lambda: SimpleNamespace(json=lambda: {})))
        def verify_identity(self): pass
        def close(self): pass
    monkeypatch.setattr(runner, 'Runtime', Resource)
    monkeypatch.setattr(runner, 'SQLiteStorage', Resource)
    monkeypatch.setattr(runner, 'LocalChatClient', Client)
    monkeypatch.setattr(runner, 'model_identity', lambda *a: {})
    monkeypatch.setattr(runner, 'gpu_competitors', lambda: [])
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a: SimpleNamespace(terminate=lambda: None))
    monkeypatch.setattr(runner, 'load_external', lambda *a: SimpleNamespace(verify_materialized=lambda *a, **k: None))
    monkeypatch.setattr('arkb.knowledge.embeddings.tokenizer_fingerprint', lambda *a: 'token-hash')
    monkeypatch.setattr(runner, 'new_observer', lambda *a: None)
    monkeypatch.setattr(runner, 'evidence_sets', lambda *a: {})
    @dataclass
    class Result:
        stop_reason: str = 'finished'
        observation: object = None
    calls = []
    def finish_then_pause(*a, **k):
        calls.append(1)
        signal.raise_signal(signal.SIGTERM)
        return Result()
    monkeypatch.setattr(runner, 'fixed_rag', finish_then_pause)
    runner.execute(tmp_path)
    status = json.loads((tmp_path / 'pilot-status.json').read_text())
    assert status['status'] == 'paused' and status['completed'] == 1
    key = attempt_key(digest(tmp_path / 'protocol.json'), schedule[0])
    completed = tmp_path / 'pilot-attempts' / key / 'complete.json'
    checksum = digest(completed)
    runner.execute(tmp_path)
    assert calls == [1] and digest(completed) == checksum
    # Explicitly clearing the administrative stop permits only unfinished work.
    (tmp_path / 'stop-request.json').unlink()
    runner.execute(tmp_path)
    assert calls == [1, 1] and digest(completed) == checksum
    assert json.loads((tmp_path / 'pilot-status.json').read_text())['completed'] == 2

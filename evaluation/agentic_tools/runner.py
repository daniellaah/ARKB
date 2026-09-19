"""Sequential, resumable registered runner; scoring labels never enter this module."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from time import monotonic, perf_counter

from arkb.config import RuntimeConfig
from arkb.runtime import Runtime
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.evaluation.external import digest, write_json, load_external
from arkb.evaluation.deadline import evaluation_deadline
from .contract import ARM_BY_ID, controlled_agent, fixed_rag, new_observer, evidence_sets
from .selection import attempt_key
from .transport import LocalChatClient, model_identity
from .common import utc
from .dependencies import verify_dependencies
from .stopping import StopController
from .readiness import run_scope_checks, operational_failures, execution_failure, call_rows


def verify_files(out, protocol):
    for name, expected in protocol['files'].items():
        if digest(out / name) != expected:
            raise ValueError('Frozen experiment input changed: ' + name)
    for name, expected in json.loads((out / 'measured-source-manifest.json').read_text()).items():
        if digest(Path(protocol['measured_source']) / name) != expected:
            raise ValueError('Measured source changed: ' + name)


NAME_PATTERNS = (' -u evaluate.py', 'mlx_lm.generate', 'mlx_lm.lora', 'benchmark_phase_c_')
OWN_MARKERS = ('agentic-retrieval-for-knowledge-bases', '/Volumes/ARKBPhaseC/', 'evaluation.agentic_tools',
               'evaluation/agentic_tools', 'evaluation.audits', 'evaluation/audits')
METAL_RECHECK_SECONDS = 300
_PS_LINE = re.compile(r'^\s*(\d+)\s+(\w{3}\s+\w{3}\s+\d+\s+\d\d:\d\d:\d\d\s+\d{4})\s+(.*)$')
_METAL_CACHE = {}


def metal_device_loaded(pid):
    """True when the process mapped the Apple GPU driver bundle (AGXMetal*)."""
    try:
        listing = subprocess.run(['/usr/sbin/lsof', '-n', '-P', '-p', str(pid)], capture_output=True,
                                 text=True, timeout=20).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    return 'AGXMetal' in listing


def gpu_competitors(*, now=None, metal_check=metal_device_loaded):
    """Named training/evaluation jobs plus any foreign Python process holding a Metal device.

    Foreign means not this project's own workers. Negative Metal checks are cached
    per process start time and rechecked after METAL_RECHECK_SECONDS; positives
    stay until the process exits. Detection yields between attempts; it never
    terminates another task.
    """
    now = monotonic() if now is None else now
    rows = subprocess.check_output(['ps', '-axo', 'pid=,lstart=,command='], text=True).splitlines()
    found, live = [], set()
    for line in rows:
        match = _PS_LINE.match(line)
        if not match:
            continue
        pid, started, command = int(match.group(1)), match.group(2), match.group(3).strip()
        if command.startswith('/bin/zsh'):
            continue
        entry = f'{pid} {command}'
        if any(p in command for p in NAME_PATTERNS):
            found.append(entry)
            continue
        if 'python' not in command.lower() or pid == os.getpid() or any(k in command for k in OWN_MARKERS):
            continue
        identity = (pid, started)
        live.add(identity)
        cached = _METAL_CACHE.get(identity)
        if cached is None or (not cached[1] and now - cached[0] >= METAL_RECHECK_SECONDS):
            cached = _METAL_CACHE[identity] = (now, bool(metal_check(pid)))
        if cached[1]:
            found.append(entry + ' [metal-device]')
    for identity in [k for k in _METAL_CACHE if k not in live]:
        del _METAL_CACHE[identity]
    return found


class CoreGuard:
    """Registered core pause rules. Failures stay outcomes; only patterns pause the run."""

    def __init__(self, policy, core_policy, *, reviewed=(), cap_hours=None):
        self.policy, self.core = policy, core_policy
        self.reviewed = set(reviewed)
        self.cap_seconds = (core_policy['trajectory_time_cap_hours'] if cap_hours is None else cap_hours) * 3600
        self.valid_calls = self.operational_failures = self.reviewed_attempts = 0
        self.measured_seconds = 0.0
        self.attempts = 0

    def observe(self, record):
        reasons = []
        self.attempts += 1
        self.measured_seconds += record['elapsed_ms'] / 1000
        key = record['key']
        if key in self.reviewed:
            self.reviewed_attempts += 1
        else:
            failures = operational_failures(record, self.policy)
            self.valid_calls += sum(not r['input_error'] for r in call_rows(record, self.policy))
            self.operational_failures += len(failures)
            if len(failures) >= self.core['max_operational_failures_per_attempt']:
                reasons.append('repeated_operational_failures:' + key)
            if record.get('competing_processes_after'):
                reasons.append('gpu_competition_during_attempt:' + key)
            elif execution_failure(record):
                reasons.append('unobserved_execution_failure:' + key)
            if (self.valid_calls >= self.core['window_min_valid_calls']
                    and self.operational_failures / self.valid_calls > self.core['window_max_operational_rate']):
                reasons.append(f'operational_failure_rate:{self.operational_failures}/{self.valid_calls}')
        if self.measured_seconds >= self.cap_seconds:
            reasons.append('trajectory_time_cap_reached')
        return reasons

    def state(self):
        return {'attempts_observed': self.attempts, 'reviewed_attempts': self.reviewed_attempts,
                'window_valid_calls': self.valid_calls, 'window_operational_failures': self.operational_failures,
                'measured_hours': self.measured_seconds / 3600, 'time_cap_hours': self.cap_seconds / 3600}


def core_guard(out, protocol, policy):
    core_policy = protocol.get('core_operational_policy')
    if protocol['phase'] != 'core' or not policy or not core_policy:
        return None
    review = out / 'operational-review.json'
    reviewed = json.loads(review.read_text())['reviewed_attempts'] if review.exists() else []
    extension = out / 'time-cap-extension.json'
    cap = json.loads(extension.read_text())['hours'] if extension.exists() else None
    return CoreGuard(policy, core_policy, reviewed=reviewed, cap_hours=cap)


def execute(out):
    with StopController(out / 'stop-request.json') as stop:
        return _execute(out, stop)


def _execute(out, stop):
    protocol = json.loads((out / 'protocol.json').read_text())
    phase = protocol['phase']
    if phase not in ('pilot', 'core') or protocol['schema'] != 'agentic-tools-v1-executable-' + phase:
        raise ValueError('Unknown registered inference protocol.')
    if phase == 'core' and protocol.get('core_dispatch_authorized') is not True:
        raise ValueError('The core protocol has not passed dispatch gates.')
    total = protocol[phase + '_attempts']
    lock = Path(protocol.get('inference_lock', out / 'inference.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid = os.getpid()
    status_path = out / f'{phase}-status.json'
    invocations = out / 'invocations'
    invocations.mkdir(exist_ok=True)
    invocation_path = invocations / (utc().replace(':', '-') + '.json')
    meta = {'status': 'running', 'stage': 'verification', 'pid': pid, 'started_at': utc(),
            'protocol_sha256': digest(out / 'protocol.json'), 'completed': 0, 'total': total}
    def status(**changes):
        meta.update(changes, updated_at=utc())
        write_json(status_path, meta)
        write_json(invocation_path, meta)
    def pause():
        if not stop.requested():
            return False
        status(status='paused', stage='between_attempts', current_attempt=None,
               administrative_stop=stop.persist())
        return True
    status()
    caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(pid)])
    client = None
    try:
        verify_files(out, protocol)
        policy = None
        if 'readiness-policy.json' in protocol['files']:
            policy = json.loads((out / 'readiness-policy.json').read_text())
            if not Path(__file__).resolve().is_relative_to(Path(protocol['measured_source']).resolve()):
                raise ValueError('Run the registered frozen runner, not the working checkout.')
        if 'dependencies.json' in protocol['files']:
            verify_dependencies(json.loads((out / 'dependencies.json').read_text()))
        guard = core_guard(out, protocol, policy)
        schedules = json.loads((out / f'{phase}-schedule.json').read_text())
        if len(schedules) != total:
            raise ValueError('Schedule size differs from the registered attempt count.')
        scenarios = json.loads((out / 'inference/scenarios.json').read_text())
        by_identity = {(r['dataset'], r['id'], r['variant']): r for r in scenarios.values()}
        inputs = json.loads((out / 'inputs.json').read_text())
        contexts = json.loads((out / 'context-indexes.json').read_text())
        attempts = out / f'{phase}-attempts'
        attempts.mkdir(exist_ok=True)
        completed = 0
        for row in schedules:
            path = attempts / attempt_key(meta['protocol_sha256'], row)
            if (path / 'result.json').exists():
                record = json.loads((path / 'result.json').read_text())
                check = json.loads((path / 'complete.json').read_text())
                if (check['result_sha256'] != digest(path / 'result.json') or record['schedule'] != row
                        or check['provider_sha256'] != digest(path / 'provider.jsonl')
                        or record['protocol_sha256'] != meta['protocol_sha256']):
                    raise ValueError('Completed attempt integrity failure.')
                completed += 1
                if guard:
                    for reason in guard.observe(record):
                        stop.request(reason)
                elif policy and (operational_failures(record, policy) or execution_failure(record)):
                    stop.request('tool_readiness_failure_in_completed_attempt')
            elif path.exists():
                raise ValueError('An interrupted attempt requires explicit accounting before resuming: ' + path.name)
        status(completed=completed, stage='prepare_runtime', **({'guard': guard.state()} if guard else {}))
        if pause():
            return
        competing = gpu_competitors()
        if competing:
            status(status='waiting_for_gpu', competing_processes=competing)
            return
        with Runtime(RuntimeConfig(offline=True, tokenizer_cache=Path(protocol['tokenizer_cache']),
                                   qdrant_url=protocol['qdrant_url'])) as runtime, ExitStack() as stack:
            tokenizer = runtime.tokenizer()
            from arkb.knowledge.embeddings import tokenizer_fingerprint
            token_hash = tokenizer_fingerprint(tokenizer)
            if token_hash != inputs['browsecomp-plus']['backend']['input']['tokenizer']:
                raise ValueError('Reference tokenizer differs from the registered snapshot.')
            counter_id = 'reference-text:' + token_hash
            counter = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
            client = LocalChatClient(protocol['models']['chat'])
            client.verify_identity()
            if model_identity(client.http, 'qwen3-embedding:0.6b') != protocol['models']['embedding']:
                raise ValueError('Embedding model identity changed.')
            active_scope, tools, engine = None, None, None
            setup_path = out / f'{phase}-setup-costs.json'
            setup = json.loads(setup_path.read_text()) if setup_path.exists() else []
            for index, row in enumerate(schedules):
                key = attempt_key(meta['protocol_sha256'], row)
                directory = attempts / key
                if (directory / 'complete.json').exists():
                    continue
                if pause():
                    return
                case = by_identity[(row['dataset'], row['id'], row['variant'])]
                scope = case['scenario_id'] if row['dataset'] == 'musique' else row['dataset']
                if scope != active_scope:
                    tools = engine = None
                    stack.close()
                    gc.collect()
                    status(stage='prepare_index_access', dataset=row['dataset'], scenario_id=case['scenario_id'])
                    start = perf_counter()
                    if row['dataset'] == 'musique':
                        entry = case
                        expected = contexts[case['scenario_id']]['build']['manifest']
                        for name, checksum in case['context_sha256'].items():
                            if digest(Path(case['corpus']) / name) != checksum:
                                raise ValueError('Context source drift.')
                        if digest(case['sqlite']) != contexts[case['scenario_id']]['sqlite_sha256']:
                            raise ValueError('Prepared context SQLite changed.')
                    else:
                        entry = inputs[row['dataset']]
                        expected = entry['index_manifest']
                        # Validation is setup work and is excluded from trial timing.
                        data = load_external(Path(entry['data']))
                        data.verify_materialized(Path(entry['corpus']), omit_empty=row['dataset'] == 'fiqa')
                        del data
                        if digest(entry['sqlite']) != entry['sqlite_sha256']:
                            raise ValueError('Preserved SQLite changed before inference.')
                    storage = stack.enter_context(SQLiteStorage(Path(entry['sqlite']), read_only=True))
                    manifest = storage.active_manifest(entry['vault_id'])
                    if asdict(manifest) != expected:
                        raise ValueError('Manifest differs from frozen protocol.')
                    engine = runtime.retrieval_engine(storage, manifest, modes=('bm25', 'semantic'), exact=True)
                    tools = runtime.agent_tools(engine=engine, directory=Path(entry['corpus']),
                                                vault_id=entry['vault_id'], rerank=False, prepare_exact=True)
                    setup.append({'scope': scope, 'elapsed_ms': (perf_counter() - start) * 1000,
                                  'index_id': manifest.index_version, 'pid': pid, 'recorded_at': utc()})
                    write_json(setup_path, setup)
                    active_scope = scope
                    # Runtime owns the cache, but release it at each scope boundary
                    # instead of retaining all prior corpora until the run ends.
                    stack.callback(tools._exact.close)
                    if pause():
                        return
                    if policy:
                        competitors = gpu_competitors()
                        if competitors:
                            status(status='waiting_for_gpu', stage='before_scope_checks', competing_processes=competitors)
                            return
                        status(stage='scope_readiness', dataset=row['dataset'], scope=scope)
                        check = run_scope_checks(tools, row['dataset'], scope, policy)
                        checks = out / 'scope-readiness'
                        checks.mkdir(exist_ok=True)
                        check_path = checks / (hashlib.sha256(scope.encode()).hexdigest() + '-' + str(pid) + '-' + str(len(setup)) + '.json')
                        check.update(protocol_sha256=meta['protocol_sha256'], index_id=manifest.index_version)
                        write_json(check_path, check)
                        manifest_path = out / 'scope-readiness-manifest.json'
                        check_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                        check_manifest[check_path.name] = digest(check_path)
                        write_json(manifest_path, check_manifest)
                        if check['status'] != 'passed':
                            stop.request('scope_readiness_failed:' + scope)
                        if pause():
                            return
                competitors = gpu_competitors()
                if competitors:
                    status(status='waiting_for_gpu', stage='between_attempts', competing_processes=competitors)
                    return
                client.verify_identity()
                if pause():
                    return
                directory.mkdir(exist_ok=False)
                write_json(directory / 'attempt.json', {'key': key, 'protocol_sha256': meta['protocol_sha256'],
                                                       'schedule': row, 'started_at': utc()})
                status(stage='inference', current_attempt=key, schedule_index=index, arm=row['arm'],
                       dataset=row['dataset'], scenario_id=case['scenario_id'])
                start = perf_counter()
                result, error = None, None
                observer = new_observer(counter, counter_id)
                with (directory / 'provider.jsonl').open('x') as journal:
                    def emit(event):
                        journal.write(json.dumps({'at': utc(), **event}, ensure_ascii=False, allow_nan=False) + '\n')
                        journal.flush()
                    client.journal = emit
                    try:
                        with evaluation_deadline(360):
                            arm = ARM_BY_ID[row['arm']]
                            fn = fixed_rag if arm.fixed else controlled_agent
                            result = fn(case['query'], tools=tools, arm=arm, client=client, observer=observer)
                    except Exception as exc:
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                client.journal = None
                elapsed_ms = (perf_counter() - start) * 1000
                record = {'key': key, 'protocol_sha256': meta['protocol_sha256'], 'schedule': row,
                          'scenario_id': case['scenario_id'], 'elapsed_ms': elapsed_ms,
                          'error': error, 'result': asdict(result) if result else None,
                          'completed_at': utc(), 'answer_quality': 'pending scoring and independent review'}
                if result:
                    record['evidence_sources'] = evidence_sets(result.observation)
                record['loaded_models_after'] = client.http.get('/api/ps').raise_for_status().json()
                record['competing_processes_after'] = gpu_competitors()
                write_json(directory / 'result.json', record)
                write_json(directory / 'complete.json', {'result_sha256': digest(directory / 'result.json'),
                                                         'provider_sha256': digest(directory / 'provider.jsonl')})
                completed += 1
                status(completed=completed, **({'guard': guard.state()} if guard else {}))
                print(json.dumps({'completed': completed, 'total': total, 'dataset': row['dataset'], 'arm': row['arm'],
                                  'stop': result.stop_reason if result else 'error', 'elapsed_ms': elapsed_ms}), flush=True)
                if guard:
                    for reason in guard.observe(record):
                        stop.request(reason)
                elif policy:
                    failures = operational_failures(record, policy)
                    if failures:
                        stop.request('tool_readiness_failed:' + key)
                    # A missing trace/hard harness failure cannot establish readiness.
                    if execution_failure(record):
                        stop.request('unobserved_execution_failure:' + key)
                    if record['competing_processes_after']:
                        stop.request('gpu_competition_during_attempt:' + key)
                if pause():
                    return
            client.verify_identity()
        verify_files(out, protocol)
        status(status='completed', stage=phase + '_inference_complete', completed=total)
    except BaseException as error:
        status(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if client:
            client.close()
        caffeinate.terminate()
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    execute(a.output.resolve())


if __name__ == '__main__':
    main()

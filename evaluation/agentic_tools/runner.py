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


class _Run:
    """Per-invocation state shared by the execution steps below."""

    def __init__(self, out, stop):
        self.out, self.stop = out, stop
        self.protocol = json.loads((out / 'protocol.json').read_text())
        self.phase = self.protocol['phase']
        if self.phase not in ('pilot', 'core') or self.protocol['schema'] != 'agentic-tools-v1-executable-' + self.phase:
            raise ValueError('Unknown registered inference protocol.')
        if self.phase == 'core' and self.protocol.get('core_dispatch_authorized') is not True:
            raise ValueError('The core protocol has not passed dispatch gates.')
        self.total = self.protocol[self.phase + '_attempts']
        self.pid = os.getpid()
        self.status_path = out / f'{self.phase}-status.json'
        invocations = out / 'invocations'
        invocations.mkdir(exist_ok=True)
        self.invocation_path = invocations / (utc().replace(':', '-') + '.json')
        self.meta = {'status': 'running', 'stage': 'verification', 'pid': self.pid, 'started_at': utc(),
                     'protocol_sha256': digest(out / 'protocol.json'), 'completed': 0, 'total': self.total}
        self.policy = self.guard = None
        self.completed = 0
        self.attempts = out / f'{self.phase}-attempts'
        self.setup_path = out / f'{self.phase}-setup-costs.json'
        self.setup = json.loads(self.setup_path.read_text()) if self.setup_path.exists() else []

    @property
    def protocol_hash(self):
        return self.meta['protocol_sha256']

    def status(self, **changes):
        if self.guard:
            changes.setdefault('guard', self.guard.state())
        self.meta.update(changes, updated_at=utc())
        write_json(self.status_path, self.meta)
        write_json(self.invocation_path, self.meta)

    def pause(self):
        """True when an administrative stop is pending; the caller returns between attempts."""
        if not self.stop.requested():
            return False
        self.status(status='paused', stage='between_attempts', current_attempt=None,
                    administrative_stop=self.stop.persist())
        return True

    def yield_for_gpu(self, stage):
        competitors = gpu_competitors()
        if competitors:
            self.status(status='waiting_for_gpu', stage=stage, competing_processes=competitors)
        return bool(competitors)

    def review(self, record, key):
        """Apply the registered pause rules to one completed or just-finished attempt."""
        if self.guard:
            for reason in self.guard.observe(record):
                self.stop.request(reason)
        elif self.policy:
            if operational_failures(record, self.policy):
                self.stop.request('tool_readiness_failed:' + key)
            # A missing trace/hard harness failure cannot establish readiness.
            if execution_failure(record):
                self.stop.request('unobserved_execution_failure:' + key)
            if record['competing_processes_after']:
                self.stop.request('gpu_competition_during_attempt:' + key)


def _verify_registration(run):
    out, protocol = run.out, run.protocol
    verify_files(out, protocol)
    if 'readiness-policy.json' in protocol['files']:
        run.policy = json.loads((out / 'readiness-policy.json').read_text())
        if not Path(__file__).resolve().is_relative_to(Path(protocol['measured_source']).resolve()):
            raise ValueError('Run the registered frozen runner, not the working checkout.')
    if 'dependencies.json' in protocol['files']:
        verify_dependencies(json.loads((out / 'dependencies.json').read_text()))
    run.guard = core_guard(out, protocol, run.policy)
    run.schedules = json.loads((out / f'{run.phase}-schedule.json').read_text())
    if len(run.schedules) != run.total:
        raise ValueError('Schedule size differs from the registered attempt count.')
    scenarios = json.loads((out / 'inference/scenarios.json').read_text())
    run.by_identity = {(r['dataset'], r['id'], r['variant']): r for r in scenarios.values()}
    run.inputs = json.loads((out / 'inputs.json').read_text())
    run.contexts = json.loads((out / 'context-indexes.json').read_text())
    run.attempts.mkdir(exist_ok=True)


def _scan_completed(run):
    """Verify every completed attempt's immutability; an interrupted one blocks resumption."""
    for row in run.schedules:
        path = run.attempts / attempt_key(run.protocol_hash, row)
        if (path / 'result.json').exists():
            record = json.loads((path / 'result.json').read_text())
            check = json.loads((path / 'complete.json').read_text())
            if (check['result_sha256'] != digest(path / 'result.json') or record['schedule'] != row
                    or check['provider_sha256'] != digest(path / 'provider.jsonl')
                    or record['protocol_sha256'] != run.protocol_hash):
                raise ValueError('Completed attempt integrity failure.')
            run.completed += 1
            if run.guard:
                run.review(record, path.name)
            elif run.policy and (operational_failures(record, run.policy) or execution_failure(record)):
                run.stop.request('tool_readiness_failure_in_completed_attempt')
        elif path.exists():
            raise ValueError('An interrupted attempt requires explicit accounting before resuming: ' + path.name)


def _prepare_scope(run, runtime, stack, row, case, scope):
    """Validate the frozen corpus/index of a scope and bind fresh tools to it; setup time is recorded apart."""
    run.status(stage='prepare_index_access', dataset=row['dataset'], scenario_id=case['scenario_id'])
    start = perf_counter()
    if row['dataset'] == 'musique':
        entry = case
        expected = run.contexts[case['scenario_id']]['build']['manifest']
        for name, checksum in case['context_sha256'].items():
            if digest(Path(case['corpus']) / name) != checksum:
                raise ValueError('Context source drift.')
        if digest(case['sqlite']) != run.contexts[case['scenario_id']]['sqlite_sha256']:
            raise ValueError('Prepared context SQLite changed.')
    else:
        entry = run.inputs[row['dataset']]
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
    run.setup.append({'scope': scope, 'elapsed_ms': (perf_counter() - start) * 1000,
                      'index_id': manifest.index_version, 'pid': run.pid, 'recorded_at': utc()})
    write_json(run.setup_path, run.setup)
    # Runtime owns the cache, but release it at each scope boundary
    # instead of retaining all prior corpora until the run ends.
    stack.callback(tools._exact.close)
    return tools, manifest


def _scope_readiness(run, tools, row, scope, manifest):
    """Run the frozen synthetic workload on the complete scope and archive its own artifact."""
    run.status(stage='scope_readiness', dataset=row['dataset'], scope=scope)
    check = run_scope_checks(tools, row['dataset'], scope, run.policy)
    checks = run.out / 'scope-readiness'
    checks.mkdir(exist_ok=True)
    check_path = checks / (hashlib.sha256(scope.encode()).hexdigest() + '-' + str(run.pid) + '-' + str(len(run.setup)) + '.json')
    check.update(protocol_sha256=run.protocol_hash, index_id=manifest.index_version)
    write_json(check_path, check)
    manifest_path = run.out / 'scope-readiness-manifest.json'
    check_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    check_manifest[check_path.name] = digest(check_path)
    write_json(manifest_path, check_manifest)
    if check['status'] != 'passed':
        run.stop.request('scope_readiness_failed:' + scope)


def _run_attempt(run, client, tools, counter, counter_id, index, row, case, key, directory):
    """Execute one registered attempt under the hard deadline and persist its immutable record."""
    directory.mkdir(exist_ok=False)
    write_json(directory / 'attempt.json', {'key': key, 'protocol_sha256': run.protocol_hash,
                                           'schedule': row, 'started_at': utc()})
    run.status(stage='inference', current_attempt=key, schedule_index=index, arm=row['arm'],
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
    record = {'key': key, 'protocol_sha256': run.protocol_hash, 'schedule': row,
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
    run.completed += 1
    run.status(completed=run.completed)
    print(json.dumps({'completed': run.completed, 'total': run.total, 'dataset': row['dataset'], 'arm': row['arm'],
                      'stop': result.stop_reason if result else 'error', 'elapsed_ms': elapsed_ms}), flush=True)
    run.review(record, key)


def _execute(out, stop):
    run = _Run(out, stop)
    lock = Path(run.protocol.get('inference_lock', out / 'inference.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.status()
    caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(run.pid)])
    client = None
    try:
        _verify_registration(run)
        _scan_completed(run)
        run.status(completed=run.completed, stage='prepare_runtime')
        if run.pause() or run.yield_for_gpu('prepare_runtime'):
            return
        protocol = run.protocol
        with Runtime(RuntimeConfig(offline=True, tokenizer_cache=Path(protocol['tokenizer_cache']),
                                   qdrant_url=protocol['qdrant_url'])) as runtime, ExitStack() as stack:
            tokenizer = runtime.tokenizer()
            from arkb.knowledge.embeddings import tokenizer_fingerprint
            token_hash = tokenizer_fingerprint(tokenizer)
            if token_hash != run.inputs['browsecomp-plus']['backend']['input']['tokenizer']:
                raise ValueError('Reference tokenizer differs from the registered snapshot.')
            counter_id = 'reference-text:' + token_hash
            counter = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
            client = LocalChatClient(protocol['models']['chat'])
            client.verify_identity()
            if model_identity(client.http, 'qwen3-embedding:0.6b') != protocol['models']['embedding']:
                raise ValueError('Embedding model identity changed.')
            active_scope, tools = None, None
            for index, row in enumerate(run.schedules):
                key = attempt_key(run.protocol_hash, row)
                directory = run.attempts / key
                if (directory / 'complete.json').exists():
                    continue
                if run.pause():
                    return
                case = run.by_identity[(row['dataset'], row['id'], row['variant'])]
                scope = case['scenario_id'] if row['dataset'] == 'musique' else row['dataset']
                if scope != active_scope:
                    tools = None
                    stack.close()
                    gc.collect()
                    tools, manifest = _prepare_scope(run, runtime, stack, row, case, scope)
                    active_scope = scope
                    if run.pause():
                        return
                    if run.policy:
                        if run.yield_for_gpu('before_scope_checks'):
                            return
                        _scope_readiness(run, tools, row, scope, manifest)
                        if run.pause():
                            return
                if run.yield_for_gpu('between_attempts'):
                    return
                client.verify_identity()
                if run.pause():
                    return
                _run_attempt(run, client, tools, counter, counter_id, index, row, case, key, directory)
                if run.pause():
                    return
            client.verify_identity()
        verify_files(out, protocol)
        run.status(status='completed', stage=run.phase + '_inference_complete', completed=run.total)
    except BaseException as error:
        run.status(status='failed', error={'type': type(error).__name__, 'message': str(error)})
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

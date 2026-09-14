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
import subprocess
import sys
from time import perf_counter

from arkb.config import RuntimeConfig
from arkb.runtime import Runtime
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.evaluation.external import digest, write_json, load_external
from arkb.evaluation.deadline import evaluation_deadline
from .contract import ARM_BY_ID, controlled_agent, fixed_rag, new_observer, evidence_sets
from .selection import attempt_key
from .transport import LocalChatClient, model_identity
from .prepare import utc
from .dependencies import verify_dependencies


def verify_files(out, protocol):
    for name, expected in protocol['files'].items():
        if digest(out / name) != expected:
            raise ValueError('Frozen experiment input changed: ' + name)
    for name, expected in json.loads((out / 'measured-source-manifest.json').read_text()).items():
        if digest(Path(protocol['measured_source']) / name) != expected:
            raise ValueError('Measured source changed: ' + name)


def gpu_competitors():
    rows = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True).splitlines()
    patterns = (' -u evaluate.py', 'mlx_lm.generate', 'mlx_lm.lora', 'benchmark_phase_c_')
    return [line.strip() for line in rows if any(p in line for p in patterns)
            and not line.strip().split(None, 1)[1].startswith('/bin/zsh')]


def execute(out):
    protocol = json.loads((out / 'protocol.json').read_text())
    phase = protocol['phase']
    if phase not in ('pilot', 'core') or protocol['schema'] != 'agentic-tools-v1-executable-' + phase:
        raise ValueError('Unknown registered inference protocol.')
    if phase == 'core' and protocol.get('core_dispatch_authorized') is not True:
        raise ValueError('The core protocol has not passed dispatch gates.')
    total = 98 if phase == 'pilot' else 5460
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
    status()
    caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(pid)])
    client = None
    try:
        verify_files(out, protocol)
        if 'dependencies.json' in protocol['files']:
            verify_dependencies(json.loads((out / 'dependencies.json').read_text()))
        competing = gpu_competitors()
        if competing:
            status(status='waiting_for_gpu', competing_processes=competing)
            return
        schedules = json.loads((out / f'{phase}-schedule.json').read_text())
        assert len(schedules) == total
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
            elif path.exists():
                raise ValueError('An interrupted attempt requires explicit accounting before resuming: ' + path.name)
        status(completed=completed, stage='prepare_runtime')
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
                competitors = gpu_competitors()
                if competitors:
                    status(status='waiting_for_gpu', stage='between_attempts', competing_processes=competitors)
                    return
                client.verify_identity()
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
                status(completed=completed)
                print(json.dumps({'completed': completed, 'total': total, 'dataset': row['dataset'], 'arm': row['arm'],
                                  'stop': result.stop_reason if result else 'error', 'elapsed_ms': elapsed_ms}), flush=True)
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

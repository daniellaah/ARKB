"""Adopt a verified full chunk cache only while the identified job is preflight.

One execution, with no retries or schedule. Preserve the original attempt and
all vectors, run the original validation once through an audited preparation
adapter, then continue the original offline replay and finalization pipeline.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

from arkb.evaluation.external import digest, write_json

ROOT = Path(__file__).resolve().parents[2]


def process(pid):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'ppid=,stat=,command='],
                            text=True, capture_output=True, check=False)
    fields = result.stdout.strip().split(maxsplit=2)
    return {'ppid': int(fields[0]), 'state': fields[1], 'command': fields[2]} if len(fields) == 3 else None


def preflight_only(run):
    if any((run / name).exists() for name in ('legs.jsonl.gz', 'rows.jsonl.gz', 'summary.json')):
        return False
    if json.loads((run / 'experiment.json').read_text())['status'] != 'indexing':
        return False
    with sqlite3.connect((run / 'index.sqlite').as_uri() + '?mode=ro', uri=True) as db:
        return all(db.execute('SELECT 1 FROM ' + name + ' LIMIT 1').fetchone() is None
                   for name in ('builds', 'snapshot_chunks', 'active_indexes'))


def check_processes(args):
    run = args.storage / 'validation/browsecomp-plus'
    expected = [(args.original_driver_pid, 1, 'resume_phase_c_prepared_once.py'),
                (args.original_coordinator_pid, args.original_driver_pid, 'complete_phase_c_validation.py'),
                (args.original_validator_pid, args.original_coordinator_pid, 'evaluation/experiments/validate_phase_c.py'),
                (args.original_worker_pid, args.original_validator_pid, str(run / 'validate_phase_c.py'))]
    observations = {}
    for pid, parent, suffix in expected:
        actual = process(pid)
        if actual is None or actual['ppid'] != parent or suffix not in actual['command'] or 'Z' in actual['state']:
            raise ValueError('Original process identity changed: ' + str(pid))
        observations[str(pid)] = actual
    service = f'gui/{os.getuid()}/{args.original_job}'
    launch = subprocess.check_output(['/bin/launchctl', 'print', service], text=True)
    matched = re.search(r'^\s*pid = (\d+)$', launch, flags=re.MULTILINE)
    if matched is None or int(matched[1]) != args.original_driver_pid:
        raise ValueError('Original launchd job identity changed.')
    return service, observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--original-job', required=True)
    parser.add_argument('--original-driver-pid', type=int, required=True)
    parser.add_argument('--original-coordinator-pid', type=int, required=True)
    parser.add_argument('--original-validator-pid', type=int, required=True)
    parser.add_argument('--original-worker-pid', type=int, required=True)
    args = parser.parse_args()
    args.storage = args.storage.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / 'status.json'
    status = {'status': 'running', 'step': 'verifying_full_recovery', 'started_at': datetime.now(timezone.utc).isoformat(),
              'original_job_stopped': False, 'plan': str(args.plan.resolve()), 'pid': os.getpid()}
    frozen = {}
    for name in ('use_phase_c_chunks.py', 'recover_phase_c_chunks.py', 'finalize_phase_c.py', Path(__file__).name):
        source = ROOT / 'evaluation/experiments' / name
        shutil.copyfile(source, args.output / name)
        frozen[name] = digest(source)
    shutil.copyfile(ROOT / 'evaluation/phase-c/v1/chunk-recovery-audit.json', args.output / 'chunk-recovery-audit.json')
    write_json(args.output / 'source-checksums.json', frozen)

    def state(step, **extra):
        status.update(step=step, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(status_path, status)

    def stage(step, command, *, env=None, log_path=None):
        state(step)
        with (log_path or args.output / (step + '.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)

    paused = False
    run = args.storage / 'validation/browsecomp-plus'
    try:
        state('verifying_full_recovery')
        plan = json.loads((args.plan / 'manifest.json').read_text())
        audit = json.loads((args.output / 'chunk-recovery-audit.json').read_text())
        if (plan['status'] != 'completed' or plan['scope'] != 'full' or plan['documents'] != 100195
                or plan['chunks'] != 1883193 or not plan['all_document_chunk_hashes_match']
                or not plan['all_embedding_inputs_match'] or audit['status'] != 'passed'
                or plan['script_sha256'] != frozen['recover_phase_c_chunks.py']
                or audit['recovery_sha256'] != frozen['recover_phase_c_chunks.py']
                or audit['adapter_sha256'] != frozen['use_phase_c_chunks.py']):
            raise ValueError('Full recovery or adapter equivalence was not verified.')
        plan_hash = digest(args.plan / 'manifest.json')
        if digest(args.plan / 'document-plans.jsonl') != plan['document_plans_sha256']:
            raise ValueError('Recovered inventory changed.')
        for name, checksum in plan['source_files'].items():
            if digest(ROOT / name) != checksum:
                raise ValueError('Prepared production source changed.')
        protocol = json.loads((run / 'protocol.json').read_text())
        for name, checksum in protocol['source_files'].items():
            if digest(run / 'measured-source' / name) != checksum:
                raise ValueError('Original measured source changed.')
        if plan['dataset_manifest_sha256'] != protocol['dataset_manifest_sha256']:
            raise ValueError('Dataset differs from original validation.')
        cache_status = json.loads((args.storage / 'cache/browsecomp-plus/status.json').read_text())
        if cache_status['status'] != 'completed':
            raise ValueError('Embedding cache is not complete.')
        completion_path = args.storage / 'validation/completion-status.json'
        completion = json.loads(completion_path.read_text())
        if (completion['status'] != 'running' or completion['jobs']['fiqa']['status'] != 'completed'
                or completion['jobs']['browsecomp-plus']['status'] != 'indexing_and_single_capture'):
            raise ValueError('Original validation workflow changed.')
        if not preflight_only(run):
            state('left_original_running', status='prepared_without_cutover', reason='Original build already advanced.')
            return
        service, observations = check_processes(args)
        # Briefly stop only the checked worker to make the preflight gate stable.
        # If it has advanced, resume it and leave its entire workflow intact.
        os.kill(args.original_worker_pid, signal.SIGSTOP)
        paused = True
        if not preflight_only(run):
            os.kill(args.original_worker_pid, signal.SIGCONT)
            paused = False
            state('left_original_running', status='prepared_without_cutover', reason='Original build advanced at cutover gate.')
            return
        _, stable = check_processes(args)
        write_json(args.output / 'cutover-gate.json', {'processes': observations, 'paused_processes': stable,
            'original_candidate_builds': 0, 'original_retrieval_captures': 0,
            'plan_sha256': plan_hash, 'audit_sha256': digest(args.output / 'chunk-recovery-audit.json'),
            'checked_at': datetime.now(timezone.utc).isoformat()})
        for source, name in ((completion_path, 'prior-completion-status.json'),
                (args.storage / 'preparation-20260912T0512/status.json', 'prior-driver-status.json'),
                (args.storage / 'validation/finalization-status.json', 'prior-finalization-status.json')):
            shutil.copyfile(source, args.output / name)
        state('stopping_verified_preflight_job')
        subprocess.run(['/bin/launchctl', 'bootout', service], check=True)
        for _ in range(40):
            if not any((p := process(int(pid))) and 'Z' not in p['state'] for pid in observations):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError('Original processes remain; refuse a second writer.')
        paused = False
        status['original_job_stopped'] = True
        state('preserving_prior_attempt')
        prior = args.output / 'prior-validation-run'
        run.rename(prior)
        run.mkdir()
        shutil.copytree(prior / 'measured-source', run / 'measured-source')
        for name in ('validate_phase_c.py', 'freeze_phase_c.py', 'run_p4.py', 'run_phase_c.py',
                     'data-manifest.json', 'input-checksums.json', 'development-decision.json', 'protocol.json'):
            shutil.copyfile(prior / name, run / name)
        state('copying_complete_existing_cache')
        # SQLite's backup API includes committed WAL pages. A raw file copy of
        # the original cache would not necessarily contain the final vectors.
        with sqlite3.connect((prior / 'index.sqlite').as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(run / 'index.sqlite') as target:
                source.backup(target, pages=16384)
                if target.execute('PRAGMA quick_check').fetchone() != ('ok',):
                    raise ValueError('Copied cache integrity check failed.')
                if target.execute('SELECT count(*) FROM embeddings').fetchone()[0] != cache_status['unique_inputs']:
                    raise ValueError('Copied cache has incomplete embeddings.')
        adaptation = {'mode': 'verified chunk/token/revision cache; original builder and retrieval',
                      'full_plan_sha256': plan_hash, 'full_plan': str(args.plan.resolve()),
                      'original_attempt': str(prior), 'original_protocol_sha256': digest(prior / 'protocol.json'),
                      'original_retrieval_captures': 0, 'scripts': frozen,
                      'audit_sha256': digest(args.output / 'chunk-recovery-audit.json'),
                      'complete_cache_backup_sha256': digest(run / 'index.sqlite')}
        write_json(run / 'prepared-index-protocol.json', adaptation)
        shutil.copyfile(args.output / 'chunk-recovery-audit.json', run / 'chunk-recovery-audit.json')
        shutil.copyfile(args.plan / 'manifest.json', run / 'recovered-chunks-manifest.json')
        for name in ('use_phase_c_chunks.py', 'recover_phase_c_chunks.py'):
            shutil.copyfile(args.output / name, run / name)
        protocol['index_preparation'] = adaptation
        write_json(run / 'protocol.json', protocol)
        completion.update(status='running', operational_recovery=str(status_path))
        write_json(completion_path, completion)
        for name in frozen:
            if digest(args.output / name) != frozen[name]:
                raise ValueError('Frozen operational adapter changed.')
        env = {**os.environ, 'PYTHONPATH': str(run / 'measured-source/src'), 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
        stage('validation', [sys.executable, '-u', str(run / 'use_phase_c_chunks.py'), '--run', str(run),
                            '--plan', str(args.plan), '--plan-sha256', plan_hash,
                            '--qdrant-url', protocol['qdrant_url']], env=env, log_path=run / 'execution.log')
        versioned = ROOT / 'evaluation/phase-c/v1/validation'
        stage('offline_replay', [sys.executable, str(ROOT / 'evaluation/audits/replay_phase_c_validation.py'),
              '--dataset', str(args.storage / 'data/browsecomp-plus'), '--run', str(run),
              '--output', str(versioned / 'browsecomp-plus-replay.json')])
        for name in ('summary', 'protocol', 'experiment', 'checksums'):
            shutil.copyfile(run / (name + '.json'), versioned / ('browsecomp-plus-' + name + '.json'))
        shutil.copyfile(args.storage / 'cache/browsecomp-plus/status.json', versioned / 'browsecomp-plus-embedding-cache.json')
        completion['jobs']['browsecomp-plus'] = {'status': 'completed', 'run': str(run),
                                                'checksums_sha256': digest(run / 'checksums.json')}
        completion.update(status='completed', updated_at=datetime.now(timezone.utc).isoformat())
        write_json(completion_path, completion)
        if digest(ROOT / 'evaluation/experiments/finalize_phase_c.py') != frozen['finalize_phase_c.py']:
            raise ValueError('Original finalizer changed.')
        stage('finalization', [sys.executable, str(ROOT / 'evaluation/experiments/finalize_phase_c.py'),
                              '--storage', str(args.storage)])
        state('completed', status='completed')
    except BaseException as error:
        state(status.get('step', 'failed'), status='failed', error={'type': type(error).__name__, 'message': str(error)})
        if status['original_job_stopped'] and 'completion' in locals() and completion.get('status') != 'completed':
            completion.update(status='failed', operational_recovery=str(status_path), error=status['error'])
            write_json(completion_path, completion)
        raise
    finally:
        if paused:
            current = process(args.original_worker_pid)
            if current and str(run / 'validate_phase_c.py') in current['command']:
                os.kill(args.original_worker_pid, signal.SIGCONT)


if __name__ == '__main__':
    main()

"""Exercise the one-shot chunk-cache handoff using fake processes and services."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from arkb.evaluation.external import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation/experiments'))
import resume_phase_c_chunks_once as workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checks = []
    cases = ('plan_mismatch', 'already_advanced', 'pid_changed', 'advanced_when_paused',
             'incomplete_cache_backup', 'normal_cutover')
    for case in cases:
        with tempfile.TemporaryDirectory(prefix='arkb-chunk-cutover-audit-') as directory:
            root = Path(directory).resolve() / 'repo'
            storage = Path(directory).resolve() / 'storage'
            experiments = root / 'evaluation/experiments'
            experiments.mkdir(parents=True)
            versioned = root / 'evaluation/phase-c/v1/validation'
            versioned.mkdir(parents=True)
            for name in ('use_phase_c_chunks.py', 'recover_phase_c_chunks.py', 'finalize_phase_c.py',
                         'resume_phase_c_chunks_once.py'):
                shutil.copyfile(ROOT / 'evaluation/experiments' / name, experiments / name)
            audit = json.loads((ROOT / 'evaluation/phase-c/v1/chunk-recovery-audit.json').read_text())
            write_json(versioned.parent / 'chunk-recovery-audit.json', audit)
            plan = storage / 'chunks'
            plan.mkdir(parents=True)
            (plan / 'document-plans.jsonl').write_text('fixture inventory\n')
            write_json(plan / 'manifest.json', {'status': 'failed' if case == 'plan_mismatch' else 'completed',
                'scope': 'full', 'documents': 100195, 'chunks': 1883193,
                'all_document_chunk_hashes_match': True, 'all_embedding_inputs_match': True,
                'script_sha256': digest(experiments / 'recover_phase_c_chunks.py'),
                'document_plans_sha256': digest(plan / 'document-plans.jsonl'),
                'source_files': {}, 'dataset_manifest_sha256': 'fixture dataset'})
            run = storage / 'validation/browsecomp-plus'
            run.mkdir(parents=True)
            (run / 'measured-source').mkdir()
            (run / 'measured-source/fixture').write_text('preserve this source')
            for name in ('validate_phase_c.py', 'freeze_phase_c.py', 'run_p4.py', 'run_phase_c.py',
                         'data-manifest.json', 'input-checksums.json', 'development-decision.json'):
                (run / name).write_text('fixture')
            write_json(run / 'protocol.json', {'source_files': {}, 'dataset_manifest_sha256': 'fixture dataset',
                                               'qdrant_url': 'http://fixture.invalid'})
            write_json(run / 'experiment.json', {'status': 'indexing'})
            with sqlite3.connect(run / 'index.sqlite') as db:
                for table in ('builds', 'snapshot_chunks', 'active_indexes', 'embeddings'):
                    db.execute('CREATE TABLE ' + table + '(value TEXT)')
                db.execute("INSERT INTO embeddings VALUES ('original vector bytes')")
                if case == 'already_advanced':
                    db.execute("INSERT INTO builds VALUES ('building')")
            original_db_hash = digest(run / 'index.sqlite')
            cache = storage / 'cache/browsecomp-plus'
            cache.mkdir(parents=True)
            write_json(cache / 'status.json', {'status': 'completed', 'unique_inputs': 2 if case == 'incomplete_cache_backup' else 1})
            write_json(storage / 'validation/completion-status.json', {'status': 'running', 'jobs': {
                'fiqa': {'status': 'completed'}, 'browsecomp-plus': {'status': 'indexing_and_single_capture'}}})
            (storage / 'preparation-20260912T0512').mkdir()
            write_json(storage / 'preparation-20260912T0512/status.json', {'status': 'running'})
            write_json(storage / 'validation/finalization-status.json', {'status': 'waiting_for_validation'})
            output = storage / 'new-job'
            events = []
            stopped = False

            def process(pid):
                if stopped:
                    return None
                parents = {101: 1, 102: 101, 103: 102, 104: 103}
                commands = {101: 'resume_phase_c_prepared_once.py', 102: 'complete_phase_c_validation.py',
                            103: 'evaluation/experiments/validate_phase_c.py', 104: str(run / 'validate_phase_c.py')}
                return {'ppid': 999 if case == 'pid_changed' else parents[pid], 'state': 'R', 'command': commands[pid]}

            def kill(pid, sig):
                events.append('pause' if sig == signal.SIGSTOP else 'resume')
                if case == 'advanced_when_paused' and sig == signal.SIGSTOP:
                    with sqlite3.connect(run / 'index.sqlite') as db:
                        db.execute("INSERT INTO builds VALUES ('just started')")

            def execute(command, **kwargs):
                nonlocal stopped
                if command[:2] == ['/bin/launchctl', 'bootout']:
                    events.append('bootout')
                    stopped = True
                else:
                    script = next(Path(value).name for value in command if str(value).endswith('.py'))
                    events.append(script)
                    if script == 'use_phase_c_chunks.py':
                        assert stopped
                        for name in ('summary', 'experiment', 'checksums'):
                            write_json(run / (name + '.json'), {'status': 'completed'})
                    elif script == 'replay_phase_c_validation.py':
                        write_json(versioned / 'browsecomp-plus-replay.json', {'status': 'passed'})
                return SimpleNamespace(returncode=0)

            argv = ['workflow', '--storage', str(storage), '--output', str(output), '--plan', str(plan),
                    '--original-job', 'fixture-job', '--original-driver-pid', '101',
                    '--original-coordinator-pid', '102', '--original-validator-pid', '103', '--original-worker-pid', '104']
            failed = False
            with ExitStack() as mocks:
                mocks.enter_context(patch.object(sys, 'argv', argv))
                mocks.enter_context(patch.object(workflow, 'ROOT', root))
                mocks.enter_context(patch.object(workflow, 'process', side_effect=process))
                mocks.enter_context(patch.object(workflow.os, 'kill', side_effect=kill))
                mocks.enter_context(patch.object(workflow.subprocess, 'run', side_effect=execute))
                mocks.enter_context(patch.object(workflow.subprocess, 'check_output', return_value='pid = 101\n'))
                try:
                    workflow.main()
                except ValueError:
                    failed = True
            status = json.loads((output / 'status.json').read_text())
            if case == 'normal_cutover':
                assert not failed and status['status'] == 'completed'
                assert events == ['pause', 'bootout', 'use_phase_c_chunks.py',
                                  'replay_phase_c_validation.py', 'finalize_phase_c.py']
            elif case in ('already_advanced', 'advanced_when_paused'):
                assert not failed and status['status'] == 'prepared_without_cutover', (case, status)
                assert events == ([] if case == 'already_advanced' else ['pause', 'resume'])
            else:
                assert failed and status['status'] == 'failed'
                assert 'use_phase_c_chunks.py' not in events
                if case != 'incomplete_cache_backup':
                    assert not events
            preserved = output / 'prior-validation-run/index.sqlite' if stopped else run / 'index.sqlite'
            if case != 'advanced_when_paused':
                assert digest(preserved) == original_db_hash
            with sqlite3.connect(preserved) as db:
                assert db.execute('SELECT value FROM embeddings').fetchone() == ('original vector bytes',)
            checks.append({'case': case, 'status': 'passed', 'events': events})
    write_json(args.output, {'status': 'passed', 'checks': checks, 'real_process_mutations': 0,
        'real_model_calls': 0, 'real_retrieval_calls': 0, 'production_writes': 0,
        'workflow_sha256': digest(workflow.__file__), 'audit_sha256': digest(__file__)})
    print(len(checks), 'cutover guard fixtures passed; no real process/service mutations.')


if __name__ == '__main__':
    main()

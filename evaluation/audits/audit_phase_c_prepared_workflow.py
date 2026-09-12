"""Exercise cutover guards with fake processes/services and temporary databases."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from arkb.evaluation.external import digest, write_json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'evaluation/experiments'))
import resume_phase_c_prepared_once as workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--frozen-helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.checkpoint.read_text())
    results = []
    for case in ('plan_mismatch', 'already_embedding', 'pid_changed', 'normal_cutover'):
        with tempfile.TemporaryDirectory(prefix='arkb-cutover-audit-') as directory:
            storage = Path(directory)
            recovery, cache = storage / 'recovery', storage / 'cache/browsecomp-plus'
            recovery.mkdir()
            cache.mkdir(parents=True)
            plan = storage / 'plans/browsecomp-plus-v1'
            plan.mkdir(parents=True)
            write_json(plan / 'manifest.json', {})
            write_json(recovery / 'cache-status.json', original)
            write_json(recovery / 'status.json', {'status': 'running', 'step': 'embedding'})
            (recovery / 'embedding.log').write_text('fixture preparation\n')
            write_json(cache / 'status.json', {'status': 'embedding' if case == 'already_embedding' else 'preparing'})
            with sqlite3.connect(cache / 'index.sqlite') as db:
                db.execute('CREATE TABLE fixture (value TEXT)')
                db.execute("INSERT INTO fixture VALUES ('preserve existing cache')")
            cache_hash = digest(cache / 'index.sqlite')
            frozen = storage / 'cache-measured-source/cache_phase_c_embeddings.py'
            frozen.parent.mkdir()
            shutil.copyfile(args.frozen_helper, frozen)
            events, stopped = [], False

            def run(command, **kwargs):
                nonlocal stopped
                if command[:2] == ['/bin/launchctl', 'bootout']:
                    events.append('bootout')
                    stopped = True
                else:
                    name = Path(command[2]).name
                    events.append(name)
                    if name == 'cache_phase_c_embeddings.py':
                        assert stopped
                        write_json(cache / 'status.json', {**original, 'status': 'completed'})
                return SimpleNamespace(returncode=0)

            def gate(*unused, **kwargs):
                events.append('verified_gate')
                if case == 'plan_mismatch':
                    raise ValueError('fixture mismatch')
                return {}, {}

            output = storage / 'new-job'
            argv = ['workflow', '--storage', str(storage), '--original-recovery', str(recovery),
                    '--output', str(output), '--original-parent-pid', '123', '--original-child-pid', '124',
                    '--original-job', 'fixture.arkb.job']
            failed = False
            with ExitStack() as mocks:
                mocks.enter_context(patch.object(sys, 'argv', argv))
                mocks.enter_context(patch.object(workflow, 'load_tokenizer', return_value=None))
                mocks.enter_context(patch.object(workflow, 'load_prepared_inputs', side_effect=gate))
                mocks.enter_context(patch.object(workflow, 'process', side_effect=lambda pid: '' if stopped else f'123 R {frozen}'))
                mocks.enter_context(patch.object(workflow.subprocess, 'run', side_effect=run))
                mocks.enter_context(patch.object(workflow.subprocess, 'check_output',
                                                return_value='pid = 999\n' if case == 'pid_changed' else 'pid = 123\n'))
                try:
                    workflow.main()
                except ValueError:
                    failed = True
            status = json.loads((output / 'status.json').read_text())
            assert digest(cache / 'index.sqlite') == cache_hash
            if case == 'normal_cutover':
                assert not failed and status['status'] == 'completed'
                assert events == ['prepare_phase_c_inputs.py', 'verified_gate', 'bootout',
                                  'cache_phase_c_embeddings.py', 'complete_phase_c_validation.py', 'finalize_phase_c.py']
                with sqlite3.connect(output / 'prior-cache.sqlite') as db:
                    assert db.execute('SELECT value FROM fixture').fetchone() == ('preserve existing cache',)
            else:
                assert 'bootout' not in events and 'cache_phase_c_embeddings.py' not in events
                assert failed == (case in ('plan_mismatch', 'pid_changed'))
                if case == 'already_embedding':
                    assert status['status'] == 'prepared_without_cutover'
            results.append({'case': case, 'status': 'passed', 'events': events})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {'status': 'passed', 'checks': results,
                            'real_process_mutations': 0, 'model_calls': 0, 'retrieval_calls': 0,
                            'production_cache_writes': 0, 'workflow_sha256': digest(workflow.__file__),
                            'audit_sha256': digest(__file__)})
    print(len(results), 'cutover guard checks passed with fake processes/services.')


if __name__ == '__main__':
    main()

"""One repaired pilot job: wait for resources, register, execute, audit, stop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.common import utc
from evaluation.agentic_tools.registration import verify_registration_inputs
from evaluation.agentic_tools.runner import gpu_competitors, verify_files
from evaluation.agentic_tools.stopping import StopController


def register_if_available(out, parent, public):
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    lock_path = Path(parent_protocol.get('inference_lock', parent / 'inference.lock'))
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'waiting_for_inference_lock'}
        competitors = gpu_competitors()
        if competitors:
            return {'status': 'waiting_for_gpu', 'competing_processes': competitors}
        if (out / 'protocol.json').exists():
            protocol = json.loads((out / 'protocol.json').read_text())
            verify_files(out, protocol)
            return {'status': 'registered', 'protocol_sha256': digest(out / 'protocol.json')}
        verify_registration_inputs(out)
        from evaluation.agentic_tools import preflight
        from evaluation.studies.agentic_pilot_v1 import freeze
        preflight.main(['--output', str(out)])
        competitors = gpu_competitors()
        if competitors:
            return {'status': 'waiting_for_gpu', 'competing_processes': competitors}
        freeze.main(['--output', str(out), '--public-dir', str(public), '--parent', str(parent), '--tool-readiness'])
        return {'status': 'registered', 'protocol_sha256': digest(out / 'protocol.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--public-dir', type=Path, required=True)
    p.add_argument('--wait-for-gpu', action='store_true')
    args = p.parse_args()
    out, parent, public = args.output.resolve(), args.parent.resolve(), args.public_dir.resolve()
    meta = {'pid': os.getpid(), 'started_at': utc(), 'output': str(out), 'core_dispatch_authorized': False}
    def status(**values):
        meta.update(values, updated_at=utc())
        write_json(out / 'pipeline-status.json', meta)
        print(json.dumps(meta), flush=True)
    # This job lock is independent of the shared GPU lock: waiting must not
    # monopolize the inference resource or allow a second copy of this pipeline.
    with (out / 'pipeline.lock').open('a') as job_lock, StopController(out / 'stop-request.json') as stop:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        try:
            while True:
                if stop.requested():
                    status(status='paused', administrative_stop=stop.persist())
                    return
                state = register_if_available(out, parent, public)
                if state != previous:
                    status(**state)
                    previous = state
                else:
                    meta['last_resource_check_at'] = utc()
                    write_json(out / 'pipeline-status.json', meta)
                if state['status'] != 'registered':
                    if not args.wait_for_gpu:
                        return
                    time.sleep(30)
                    continue
                if stop.requested():
                    continue
                source = out / 'measured-source'
                env = {**os.environ, 'PYTHONPATH': str(source / 'src') + ':' + str(source),
                       'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
                status(status='running_pilot')
                worker = subprocess.Popen([sys.executable, '-u', '-m', 'evaluation.agentic_tools.runner', '--output', str(out)],
                                          env=env, cwd=source)
                status(worker_pid=worker.pid)
                while worker.poll() is None:
                    if stop.requested():
                        # The child's stop controller observes the same file.
                        stop.persist()
                    time.sleep(1)
                if worker.returncode:
                    raise RuntimeError('Registered pilot worker failed with exit code ' + str(worker.returncode))
                result = json.loads((out / 'pilot-status.json').read_text())
                if result['status'] == 'waiting_for_gpu' and args.wait_for_gpu:
                    previous = None
                    continue
                subprocess.run([sys.executable, '-m', 'evaluation.agentic_tools.summarize_pilot', '--output', str(out)],
                               env=env, cwd=source, check=True)
                gate = json.loads((out / 'tool-readiness.json').read_text())
                status(status='pilot_complete' if result['status'] == 'completed' else result['status'],
                       completed=result['completed'], total=result['total'], tool_readiness=gate['status'],
                       next_step='review pilot, register concurrency/repeatability workload, justify replacement core design')
                return
        except BaseException as exc:
            status(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
            raise


if __name__ == '__main__':
    main()

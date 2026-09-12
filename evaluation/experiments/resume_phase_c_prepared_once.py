"""Prepare full inputs once, then replace only the identified serial preparation.

The original job continues until the complete input plan matches its saved
serial digest. If it has already advanced, leave it running and keep the plan
for future recovery. No scheduled retries, retrieval repetition or policy tuning.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time

from arkb.evaluation.external import digest, write_json
from arkb.knowledge.embeddings import load_tokenizer
from arkb.knowledge.models import EmbeddingSpec
from cache_phase_c_embeddings import load_prepared_inputs


def process(pid):
    result = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'ppid=,stat=,command='],
                            text=True, capture_output=True, check=False)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--original-recovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original-parent-pid', type=int, required=True)
    parser.add_argument('--original-child-pid', type=int, required=True)
    parser.add_argument('--original-job', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    plan = args.storage / 'plans/browsecomp-plus-v1'
    cache = args.storage / 'cache/browsecomp-plus'
    dataset = args.storage / 'data/browsecomp-plus'
    original_path = args.original_recovery / 'cache-status.json'
    original = json.loads(original_path.read_text())
    scripts = ('prepare_phase_c_inputs.py', 'cache_phase_c_embeddings.py',
               'complete_phase_c_validation.py', 'finalize_phase_c.py', Path(__file__).name)
    source_hashes = {}
    for name in scripts:
        source = root / 'evaluation/experiments' / name
        shutil.copyfile(source, args.output / name)
        source_hashes[name] = digest(source)
    write_json(args.output / 'source-checksums.json', source_hashes)
    status = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'mode': 'one execution; exact full input gate; original model and ordered cache writer',
              'original_job': args.original_job, 'plan': str(plan), 'original_job_stopped': False}

    def stage(name, script, parameters):
        if digest(root / 'evaluation/experiments' / script) != source_hashes[script]:
            raise ValueError('Operational source changed after launch: ' + script)
        status.update(step=name, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output / 'status.json', status)
        with (args.output / (name + '.log')).open('x') as log:
            subprocess.run([sys.executable, '-u', str(root / 'evaluation/experiments' / script), *parameters],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        stage('preparation', 'prepare_phase_c_inputs.py', [
            '--dataset', str(dataset), '--expected-checkpoint', str(original_path),
            '--output', str(plan), '--workers', '8'])
        status.update(step='verifying_plan_before_cutover')
        write_json(args.output / 'status.json', status)
        tokenizer = load_tokenizer(cache_dir=root / '.uv-cache/tokenizers', local_files_only=True)
        _, inputs = load_prepared_inputs(plan, dataset=dataset, tokenizer=tokenizer,
                                        spec=EmbeddingSpec(**original['embedding_spec']))
        del inputs, tokenizer
        status['verified_plan_sha256'] = digest(plan / 'manifest.json')
        state = json.loads((cache / 'status.json').read_text())
        recovery = json.loads((args.original_recovery / 'status.json').read_text())
        if state['status'] in ('embedding', 'completed') and recovery['status'] in ('running', 'completed'):
            status.update(status='prepared_without_cutover', reason='Original job already advanced; left untouched.')
            return
        if state['status'] != 'preparing' or recovery.get('step') != 'embedding' or recovery['status'] != 'running':
            raise ValueError('Original workflow state requires inspection; refuse cutover.')
        if (args.storage / 'validation/browsecomp-plus').exists():
            raise FileExistsError('A retrieval capture already exists; do not replace its workflow.')
        service = f'gui/{os.getuid()}/{args.original_job}'
        launch = subprocess.check_output(['/bin/launchctl', 'print', service], text=True)
        pid = re.search(r'^\s*pid = (\d+)$', launch, flags=re.MULTILINE)
        child = process(args.original_child_pid).split(maxsplit=2)
        frozen_helper = args.storage / 'cache-measured-source/cache_phase_c_embeddings.py'
        if (pid is None or int(pid[1]) != args.original_parent_pid or len(child) != 3
                or int(child[0]) != args.original_parent_pid or str(frozen_helper) not in child[2]
                or digest(frozen_helper) != original['script_sha256']):
            raise ValueError('Original job/process identity changed; refuse cutover.')
        # The exact job and child are still in preparation. Cache transactions are
        # durable; do not open a new writer until all its processes are absent.
        subprocess.run(['/bin/launchctl', 'bootout', service], check=True)
        for _ in range(20):
            remaining = [process(pid) for pid in (args.original_parent_pid, args.original_child_pid)]
            if not any(remaining):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('Original process remains; refuse a second writer.')
        status['original_job_stopped'] = True
        status['step'] = 'preserving_cutover_checkpoint'
        write_json(args.output / 'status.json', status)
        for source, name in ((cache / 'status.json', 'prior-cache-status.json'),
                             (args.original_recovery / 'status.json', 'prior-recovery-status.json'),
                             (args.original_recovery / 'embedding.log', 'prior-embedding.log')):
            shutil.copyfile(source, args.output / name)
        with sqlite3.connect(cache.joinpath('index.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(args.output / 'prior-cache.sqlite') as target:
                source.backup(target)
                if target.execute('PRAGMA quick_check').fetchone() != ('ok',):
                    raise ValueError('Preserved cache integrity check failed.')
        write_json(args.output / 'prior-checksums.json', {
            p.name: digest(p) for p in args.output.glob('prior-*')})
        stage('embedding', 'cache_phase_c_embeddings.py', [
            '--dataset', str(dataset), '--output', str(cache), '--prepared-plan', str(plan)])
        completed = json.loads((cache / 'status.json').read_text())
        for key in ('documents', 'chunks', 'unique_inputs', 'input_sha256', 'tokenizer', 'embedding_spec',
                    'dataset_manifest_sha256', 'batch_size', 'max_batch_tokens', 'chunk_size', 'chunk_overlap'):
            if completed[key] != original[key]:
                raise ValueError('Completed cache differs from original configuration: ' + key)
        stage('validation', 'complete_phase_c_validation.py', [
            '--storage', str(args.storage), '--qdrant-url', 'http://127.0.0.1:6340', '--resume'])
        stage('finalization', 'finalize_phase_c.py', ['--storage', str(args.storage)])
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'status.json', status)


if __name__ == '__main__':
    main()

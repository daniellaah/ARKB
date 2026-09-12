"""Resume the interrupted Phase C computation once, preserving completed captures.

Run under a one-shot OS job (no calendar, repeat interval or automatic retry),
so the computation does not depend on an interactive tool session staying alive.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from arkb.evaluation.external import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.recovery.mkdir(parents=True, exist_ok=False)
    cache = args.storage / 'cache/browsecomp-plus'
    frozen = args.storage / 'cache-measured-source'
    prior = json.loads((cache / 'status.json').read_text())
    if prior['status'] != 'embedding':
        raise ValueError('This recovery requires the preserved interrupted embedding checkpoint.')
    if digest(frozen / 'cache_phase_c_embeddings.py') != prior['script_sha256']:
        raise ValueError('Original embedding helper changed.')
    if (args.storage / 'validation/browsecomp-plus').exists():
        raise FileExistsError('Do not repeat or overwrite a retrieval capture.')

    # No cache writer is running at launch. Preserve the interrupted DB and WAL
    # together before reopening the existing durable cache through normal code.
    evidence = {
        'cache-status.json': cache / 'status.json',
        'validation-status.json': args.storage / 'validation/completion-status.json',
        'finalization-status.json': args.storage / 'validation/finalization-status.json',
        'cache.log': root / 'evaluation/results/phase-c-v1/cache-browsecomp.log',
        'validation.log': root / 'evaluation/results/phase-c-v1/validation-completion.log',
        'finalization.log': root / 'evaluation/results/phase-c-v1/finalization-post-scale.log',
        'cache-integrity.json': root / 'evaluation/results/phase-c-v1/interruption-cache-check.json',
        'driver.py': Path(__file__),
        'validation-driver.py': root / 'evaluation/experiments/complete_phase_c_validation.py',
    }
    for path in cache.glob('index.sqlite*'):
        evidence['interrupted-' + path.name] = path
    for name, path in evidence.items():
        shutil.copyfile(path, args.recovery / name)
    write_json(args.recovery / 'prior-checksums.json', {
        name: digest(args.recovery / name) for name in evidence
    })
    status = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'reason': 'The previous three workers disappeared without final status or a Python traceback; exact cause is undetermined.',
              'mode': 'one execution; durable cache reuse; no repeated retrieval, tuning or automatic retries'}
    steps = [
        ('embedding', [sys.executable, '-u', str(frozen / 'cache_phase_c_embeddings.py'),
                       '--dataset', str(args.storage / 'data/browsecomp-plus'), '--output', str(cache)]),
        ('validation', [sys.executable, '-u', str(root / 'evaluation/experiments/complete_phase_c_validation.py'),
                        '--storage', str(args.storage), '--qdrant-url', 'http://127.0.0.1:6340', '--resume']),
        ('finalization', [sys.executable, '-u', str(root / 'evaluation/experiments/finalize_phase_c.py'),
                          '--storage', str(args.storage)]),
    ]
    try:
        for name, command in steps:
            status.update(step=name, updated_at=datetime.now(timezone.utc).isoformat())
            write_json(args.recovery / 'status.json', status)
            print(name, 'starting', flush=True)
            with (args.recovery / (name + '.log')).open('x') as log:
                env = dict(os.environ)
                if name == 'embedding':
                    env['PYTHONPATH'] = str(frozen / 'src')
                subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            if name == 'embedding':
                completed = json.loads((cache / 'status.json').read_text())
                for key in ('dataset_manifest_sha256', 'script_sha256', 'batch_size', 'max_batch_tokens',
                            'chunk_size', 'chunk_overlap', 'embedding_spec', 'tokenizer', 'documents',
                            'chunks', 'unique_inputs', 'input_sha256'):
                    if completed[key] != prior[key]:
                        raise ValueError('Embedding preparation drift after recovery: ' + key)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(args.recovery / 'status.json', status)


if __name__ == '__main__':
    main()

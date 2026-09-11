"""Verify portable Phase B parts and reproduce every sealed score analysis offline."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from arkb.evaluation.external import digest, write_json


def local_file(folder, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Expected an artifact basename.')
    path = folder/name
    if path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular artifact file.')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Retain previous replay evidence; choose a new output path.')
    parts = json.loads(args.manifest.read_text()); folder = args.manifest.parent
    if parts['schema'] != 'arkb-phase-b-multipart-v1' or Path(parts['root']).name != parts['root']:
        raise ValueError('Unknown or invalid archive identity.')
    manifest_path = local_file(folder, parts['archive_manifest'])
    if digest(manifest_path) != parts['archive_manifest_sha256']:
        raise ValueError('Archive file manifest changed.')
    manifest = json.loads(manifest_path.read_text())
    names = [p['name'] for p in parts['parts']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Missing or duplicated transport parts.')
    matched = {}
    with tempfile.TemporaryDirectory(prefix='arkb-phase-b-replay-') as temporary:
        temp = Path(temporary).resolve(); archive = temp/'experiments.tar.gz'
        with archive.open('xb') as target:
            for part in parts['parts']:
                path = local_file(folder, part['name'])
                if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
                    raise ValueError('Part bytes changed: ' + part['name'])
                with path.open('rb') as source:
                    shutil.copyfileobj(source, target)
        if archive.stat().st_size != parts['archive_bytes'] or digest(archive) != parts['archive_sha256']:
            raise ValueError('Reassembled archive differs from the sealed original.')
        if (manifest['archive_sha256'], manifest['archive_bytes']) != (parts['archive_sha256'], parts['archive_bytes']):
            raise ValueError('Archive manifests disagree.')
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            expected = {parts['root']+'/'+n for n in manifest['files']} | {parts['root']+'/archive-manifest.json'}
            if (len(members) != len(expected) or {m.name for m in members} != expected
                    or not all(m.isfile() for m in members)):
                raise ValueError('Archive member set differs.')
            stream.extractall(temp, filter='data')
        run = temp/parts['root']
        embedded = json.loads((run/'archive-manifest.json').read_text())
        if embedded['files'] != manifest['files']:
            raise ValueError('Embedded file manifest differs.')
        for name, expected_hash in manifest['files'].items():
            path = (run/name).resolve()
            if not path.is_relative_to(run) or digest(path) != expected_hash:
                raise ValueError('Extracted file changed: ' + name)
        env = {**os.environ, 'PYTHONPATH': str(run/'B5/measured-source/src'),
               'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}
        def execute(script, output, *extra):
            completed = subprocess.run([sys.executable, str(script), str(run), '--output', str(output), *extra],
                                       env=env, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(completed.stderr or completed.stdout)
        for variant, arm in (('B0', 'B0-inference'), ('B1', 'B1'), ('B2', 'B2-corrected'),
                             ('B3', 'B3'), ('B4', 'B4'), ('B5', 'B5')):
            original = run/'final-analyses'/variant
            output = temp/'analyses'/variant
            execute(original/'audit.py', output, '--variant', arm)
            if digest(original/'summary.json') != digest(output/'summary.json'):
                raise ValueError('Offline analysis does not reproduce: ' + variant)
            matched[variant] = digest(output/'summary.json')
            print('reproduced', variant, flush=True)
        execute(run/'selection/select.py', temp/'selection')
        if digest(temp/'selection/decision.json') != digest(run/'selection/decision.json'):
            raise ValueError('Registered policy selection differs.')
        execute(run/'diagnostic-scripts/verify_phase_b_boundary.py', temp/'boundary')
        for name in ('summary.json', 'scifact.jsonl', 'bright-stackoverflow.jsonl', 'bright-robotics.jsonl'):
            if digest(temp/'boundary'/name) != digest(run/'boundary-audit'/name):
                raise ValueError('Frozen Hybrid boundary replay differs: ' + name)
    result = {'schema': 'arkb-phase-b-offline-replay-v1', 'status': 'completed',
              'parts_verified': len(names), 'archive_byte_identity': True,
              'archive_sha256': parts['archive_sha256'], 'files_verified': len(manifest['files']),
              'analysis_summaries_byte_identical': matched, 'selection_byte_identical': True,
              'hybrid_boundary_byte_identical': True, 'new_model_calls': 0,
              'upstream_retrieval_calls': 0, 'replayer_sha256': digest(Path(__file__)),
              'release_eligible': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

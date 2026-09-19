"""Prepare a new development pilot without changing parent or corpus artifacts."""
import argparse
import json
from pathlib import Path
import shutil

from arkb.evaluation.external import digest, write_json
from .prepare import ROOT, utc
from .readiness import POLICY
from .runner import verify_files


def prepare(parent, out):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    protocol = json.loads((parent / 'protocol.json').read_text())
    if json.loads((parent / 'pilot-status.json').read_text())['status'] != 'completed':
        raise ValueError('The parent development pilot must be complete.')
    verify_files(parent, protocol)
    repair = ROOT / 'evaluation/agentic-tools/tool-overhead-v1'
    manifest = json.loads((repair / 'repair-source-manifest.json').read_text())
    verified = {name: value for name, value in manifest.items() if name.startswith('src/')}
    if not verified or any(digest(ROOT / name) != value for name, value in verified.items()):
        raise ValueError('Production code differs from the verified tool repair.')
    summary = json.loads((repair / 'summary.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    names = ['selection.json', 'design.json', 'inputs.json', 'context-indexes.json',
             'inference/scenarios.json', 'pilot-schedule.json', 'core-schedule.json',
             'judge-design.json', 'dependencies.json', 'scoring/browsecomp-answers.json',
             'scoring/musique.json', 'scoring/browsecomp-plus.json', 'scoring/fiqa.json', 'scoring/nfcorpus.json']
    for name in names:
        if digest(parent / name) != protocol['files'][name]:
            raise ValueError('Parent input changed: ' + name)
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, out / name)
    design = json.loads((out / 'design.json').read_text())
    design.update(status='repaired_development_pilot_v3',
                  note='Same 98 development trajectories; repaired tools and preregistered operational gates. Copied core schedule is historical capacity only.')
    write_json(out / 'design.json', design)
    shutil.copyfile(parent / 'pilot-accounting.json', out / 'parent-pilot-accounting.json')
    shutil.copyfile(ROOT / 'docs/agentic-tool-selection-pilot-amendment-v3.md', out / 'amendment.md')
    write_json(out / 'readiness-policy.json', POLICY)
    write_json(out / 'repair-verification.json', {'summary': summary, 'summary_sha256': digest(repair / 'summary.json'),
                                                'production_files': verified})
    write_json(out / 'preparation-provenance.json', {
        'prepared_at': utc(), 'parent_protocol_sha256': digest(parent / 'protocol.json'),
        'parent_accounting_sha256': digest(parent / 'pilot-accounting.json'),
        'copied_inputs': {name: digest(parent / name) for name in names},
        'selection': 'All original development scenarios; no answer-based reselection',
        'data_and_index_mutations': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    prepare(args.parent, args.output)
    print(json.dumps({'status': 'prepared_not_frozen', 'output': str(args.output)}))


if __name__ == '__main__':
    main()

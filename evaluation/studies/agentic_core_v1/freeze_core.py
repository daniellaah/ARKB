"""Freeze core inputs only after complete pilot and qualified calibration review."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import httpx

from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.contract import OPTIONS, THINK
from evaluation.agentic_tools.dependencies import verify_dependencies
from evaluation.agentic_tools.common import ROOT, utc
from evaluation.agentic_tools.records import read_records
from evaluation.agentic_tools.runner import verify_files, gpu_competitors
from evaluation.agentic_tools.selection import schedule
from evaluation.agentic_tools.summarize_pilot import audit_record
from evaluation.agentic_tools.transport import model_identity


def freeze(pilot, calibration, output, public):
    parent, rows = read_records(pilot)
    verify_files(pilot, parent)
    review = json.loads((pilot / 'technical-review.json').read_text())
    accounting = json.loads((pilot / 'pilot-accounting.json').read_text())
    if (parent['phase'] != 'pilot' or len(rows) != 98 or review['status'] != 'passed_for_core_preparation'
            or review['protocol_sha256'] != digest(pilot / 'protocol.json')
            or review['accounting_sha256'] != digest(pilot / 'pilot-accounting.json')
            or accounting['status'] != 'complete'):
        raise ValueError('Complete reviewed pilot is required.')
    design = json.loads((pilot / 'design.json').read_text())
    if design['think'] is not THINK or design['options'] != OPTIONS:
        raise ValueError('Core settings must equal the reviewed pilot.')
    checks = sum(audit_record(r, options=OPTIONS, think=THINK)['checks'] for r in rows)
    if checks != review['trace_checks']:
        raise ValueError('Pilot replay count changed.')
    cal_protocol = json.loads((calibration / 'protocol.json').read_text())
    cal_summary = json.loads((calibration / 'summary.json').read_text())
    controls = json.loads((calibration / 'controls.json').read_text())
    if (cal_protocol['source_protocol_sha256'] != digest(pilot / 'protocol.json')
            or cal_summary['responses'] != 42 or cal_summary['status'] != 'completed'
            or cal_summary['pending_scores'] != 0 or len(controls) != 2
            or any(c['correct'] is not c['expected'] for c in controls)):
        raise ValueError('Calibration execution and synthetic controls are incomplete.')
    for name in ('judge.py', 'labels.py', 'scoring.py'):
        if digest(ROOT / 'evaluation/agentic_tools' / name) != cal_protocol['source_sha256'][name]:
            raise ValueError('Scoring behavior changed after calibration: ' + name)
    original_sources = json.loads((pilot / 'measured-source-manifest.json').read_text())
    guarded = {name: sha for name, sha in original_sources.items()
               if name.startswith('src/') or name in ('evaluation/agentic_tools/contract.py', 'evaluation/agentic_tools/transport.py')}
    for name, expected in guarded.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Measured inference behavior differs from v2: ' + name)
    dependencies = json.loads((pilot / 'dependencies.json').read_text())
    verify_dependencies(dependencies)
    if gpu_competitors():
        raise ValueError('Competing GPU work prevents core freeze/dispatch.')
    with httpx.Client(base_url='http://127.0.0.1:11434', timeout=30) as http:
        if http.get('/api/version').raise_for_status().json() != parent['ollama_version']:
            raise ValueError('Model server changed after pilot.')
        for identity in parent['models'].values():
            if model_identity(http, identity['name']) != identity:
                raise ValueError('Model identity changed.')
    tests = ET.parse(public / 'boundary-tests.xml').getroot()
    suites = [tests] if tests.tag == 'testsuite' else list(tests)
    if sum(int(s.get('tests', 0)) for s in suites) < 46 or any(int(s.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Core boundary/scoring/statistics checks are incomplete.')
    selection = json.loads((pilot / 'selection.json').read_text())
    core_schedule = json.loads((pilot / 'core-schedule.json').read_text())
    if core_schedule != schedule(selection, 'core') or len(core_schedule) != 5460:
        raise ValueError('Original core schedule changed.')
    output.mkdir(exist_ok=False)
    inputs = ['selection.json', 'design.json', 'inputs.json', 'context-indexes.json',
              'inference/scenarios.json', 'core-schedule.json', 'provider-preflight.json',
              'judge-design.json', 'dependencies.json', 'scoring/browsecomp-answers.json',
              'scoring/browsecomp-plus.json', 'scoring/musique.json', 'scoring/fiqa.json', 'scoring/nfcorpus.json']
    for name in inputs:
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pilot / name, dest)
    for origin, name in [(pilot / 'technical-review.json', 'pilot-technical-review.json'),
                         (pilot / 'pilot-accounting.json', 'pilot-accounting.json'),
                         (calibration / 'summary.json', 'calibration-summary.json'),
                         (calibration / 'protocol.json', 'calibration-protocol.json'),
                         (calibration / 'controls.json', 'calibration-controls.json'),
                         (public / 'boundary-tests.xml', 'boundary-tests.xml'),
                         (ROOT / 'docs/agentic-tool-selection-core-protocol.md', 'core-protocol.md'),
                         (ROOT / 'docs/agentic-tool-selection-pilot-amendment-v2.md', 'pilot-amendment.md')]:
        shutil.copyfile(origin, output / name)
        inputs.append(name)
    source = output / 'measured-source'
    source.mkdir()
    shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'evaluation/agentic_tools', source / 'evaluation/agentic_tools',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    source_files = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    write_json(output / 'measured-source-manifest.json', source_files)
    inputs.append('measured-source-manifest.json')
    protocol = {**parent, 'schema': 'agentic-tools-v1-executable-core', 'phase': 'core', 'created_at': utc(),
                'output': str(output), 'public_dir': str(public), 'measured_source': str(source),
                'files': {name: digest(output / name) for name in inputs},
                'parent_output': str(pilot), 'parent_protocol_sha256': digest(pilot / 'protocol.json'),
                'core_dispatch_authorized': True,
                'core_gate': 'Complete v2 and 1184 trace checks; unchanged inference behavior; 42 automatic calibration records with independent review explicitly pending; frozen scoring and statistics.',
                'guarded_inference_files': guarded, 'calibration_output': str(calibration),
                'quality_status': 'provisional_pending_independent_human_review',
                'cost_estimate': {'core_inference_hours': accounting['projected_core_inference_hours'],
                                 'core_attempt_artifact_gib': accounting['projected_core_artifact_gib'],
                                 'excludes': accounting['estimate_excludes']},
                'statistics': {'seed': 20260912, 'resamples': 20000, 'primary_contrasts': ['A-M', 'A-B', 'A-S', 'A-H'],
                               'nominal_percentiles': [2.5, 97.5], 'primary_adjusted_percentiles': [.625, 99.375],
                               'minimum_gain': .05, 'unit': 'query; MuSiQue original pair stratified by hop'},
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    # Parent amendment path belongs to the pilot; retain its identity explicitly.
    protocol.pop('amendment_sha256', None)
    write_json(output / 'protocol.json', protocol)
    write_json(public / 'protocol.json', protocol)
    print(json.dumps({'core_protocol_sha256': digest(output / 'protocol.json'), 'source_files': len(source_files),
                      'guarded_inference_files': len(guarded), 'core_attempts': 5460}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--public', type=Path, required=True)
    a = p.parse_args()
    freeze(a.pilot.resolve(), a.calibration.resolve(), a.output.resolve(), a.public.resolve())


if __name__ == '__main__':
    main()

"""Replacement core v2: gated preparation, resource-aware registration, execution and accounting.

Answer labels and judge scores never influence anything in this module. The
registered inference behavior is the unchanged pilot-v3 source; only harness
bookkeeping (pause rules, time cap, competitor detection) is new.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import httpx
import numpy as np

from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.contract import OPTIONS, THINK
from .design import (CORE_OPERATIONAL_POLICY, COUNTS, PRIMARY_REPETITIONS, PRIMARY_TOTAL, REPEAT_SUBSET,
                          REPEAT_TOTAL, TOTAL, core_schedule, exposed_questions, repeat_ids, selection_for_core)
from evaluation.agentic_tools.dependencies import verify_dependencies
from evaluation.agentic_tools.registration import verify_registration_inputs, working_files
from evaluation.agentic_tools.common import ROOT, utc
from evaluation.agentic_tools.readiness import POLICY, assess
from evaluation.agentic_tools.records import read_records
from evaluation.agentic_tools.runner import CoreGuard, gpu_competitors, verify_files
from evaluation.agentic_tools.stopping import StopController
from evaluation.agentic_tools.summarize_pilot import audit_record, failure_category
from evaluation.agentic_tools.transport import model_identity

INFERENCE_FILES = ['evaluation/agentic_tools/' + n for n in
                   ('contract.py', 'transport.py', 'dependencies.py', 'selection.py', 'readiness.py', 'stopping.py')]
SCORING_FILES = ('judge.py', 'labels.py', 'scoring.py')
LOADER_FILES = ('records.py',)
COPIED_INPUTS = ['inputs.json', 'context-indexes.json', 'judge-design.json', 'dependencies.json', 'readiness-policy.json',
                 'scoring/browsecomp-answers.json', 'scoring/musique.json', 'scoring/browsecomp-plus.json',
                 'scoring/fiqa.json', 'scoring/nfcorpus.json']
STATISTICS = {'seed': 20260912, 'resamples': 20000, 'primary_contrasts': ['A-M', 'A-B', 'A-S', 'A-H'],
              'nominal_percentiles': [2.5, 97.5], 'primary_adjusted_percentiles': [.625, 99.375],
              'minimum_gain': .05, 'unit': 'query; MuSiQue original pair stratified by hop'}
REVISION = 'replacement-core-v2'
MINIMUM_TESTS = 1290
MINIMUM_FREE_GIB = 10


def load(path):
    return json.loads(Path(path).read_text())


def check_tests(path, minimum=MINIMUM_TESTS):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root)
    passed = sum(int(s.get('tests', 0)) for s in suites)
    if passed < minimum or any(int(s.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped')):
        raise ValueError('Regression tests are incomplete or failing.')
    return passed


def inference_files_equal(pilot):
    manifest = load(pilot / 'measured-source-manifest.json')
    names = [n for n in manifest if n.startswith('src/')] + INFERENCE_FILES
    for name in names:
        if digest(ROOT / name) != manifest[name]:
            raise ValueError('Measured inference behavior differs from pilot v3: ' + name)
    return {n: manifest[n] for n in names}


def gate_evidence(pilot, old_core, repeat, serving, calibration, design_dir):
    """Every dispatch gate, checked from immutable artifacts; no answer scores are read."""
    pilot_protocol, pilot_rows = read_records(pilot)
    verify_files(pilot, pilot_protocol)
    pilot_hash = digest(pilot / 'protocol.json')
    status = load(pilot / 'pilot-status.json')
    readiness = load(pilot / 'tool-readiness.json')
    accounting = load(pilot / 'pilot-accounting.json')
    pilot_audit = load(ROOT / 'evaluation/agentic-tools/v3/completion-audit.json')
    if (pilot_protocol['phase'] != 'pilot' or pilot_protocol.get('revision') != 'tool-repair-pilot-v3'
            or len(pilot_rows) != 98 or status['status'] != 'completed' or status['completed'] != 98
            or readiness['status'] != 'passed' or readiness['protocol_sha256'] != pilot_hash
            or accounting['status'] != 'complete' or accounting['tool_readiness'] != 'passed'
            or not pilot_audit['status'].startswith('passed') or pilot_audit['protocol_sha256'] != pilot_hash
            or pilot_audit['attempts'] != 98 or pilot_audit['operational_failures'] != 0):
        raise ValueError('Complete, audited, tool-ready pilot v3 is required.')
    if load(pilot / 'readiness-policy.json') != POLICY:
        raise ValueError('Frozen readiness policy differs from the executable policy.')
    design = load(pilot / 'design.json')
    if design['options'] != OPTIONS or design['think'] is not THINK:
        raise ValueError('Core settings must equal pilot v3.')
    repeat_protocol = load(repeat / 'protocol.json')
    repeat_status = load(repeat / 'pilot-status.json')
    repeat_summary = load(repeat / 'repeatability-summary.json')
    repeat_audit = load(ROOT / 'evaluation/agentic-tools/repeatability-v1/completion-audit.json')
    repeat_hash = digest(repeat / 'protocol.json')
    if (repeat_protocol.get('revision') != 'full-agent-repeatability-v1' or repeat_protocol['parent_protocol_sha256'] != pilot_hash
            or repeat_status['status'] != 'completed' or repeat_status['completed'] != 70
            or repeat_summary['status'] != 'completed' or repeat_summary['protocol_sha256'] != repeat_hash
            or repeat_audit['status'] != 'passed' or repeat_audit['protocol_sha256'] != repeat_hash):
        raise ValueError('Completed and audited repeatability-v1 is required.')
    screen = load(serving / 'summary.json')
    serving_audit = load(ROOT / 'evaluation/agentic-tools/serving-v1/completion-audit.json')
    serving_hash = digest(serving / 'protocol.json')
    if (screen['status'] != 'completed' or any(g['eligible_for_agent_confirmation'] for g in screen['groups'])
            or serving_audit['status'] != 'passed' or serving_audit['protocol_sha256'] != serving_hash
            or screen['protocol_sha256'] != serving_hash):
        raise ValueError('The single-slot serving policy requires the completed negative concurrency screen.')
    cal_protocol = load(calibration / 'protocol.json')
    cal_summary = load(calibration / 'summary.json')
    controls = load(calibration / 'controls.json')
    if (cal_summary['responses'] != 42 or cal_summary['status'] != 'completed' or cal_summary['pending_scores'] != 0
            or len(controls) != 2 or any(c['correct'] is not c['expected'] for c in controls)):
        raise ValueError('Calibration execution and synthetic controls are incomplete.')
    for name in SCORING_FILES:
        if digest(ROOT / 'evaluation/agentic_tools' / name) != cal_protocol['source_sha256'][name]:
            raise ValueError('Scoring behavior changed after calibration: ' + name)
    pilot_manifest = load(pilot / 'measured-source-manifest.json')
    for name in LOADER_FILES:
        if digest(ROOT / 'evaluation/agentic_tools' / name) != pilot_manifest['evaluation/agentic_tools/' + name]:
            raise ValueError('Attempt loader differs from the v3 snapshot: ' + name)
    judge_design = load(pilot / 'judge-design.json')
    if digest(ROOT / 'evaluation/agentic_tools/labels.py') != judge_design['parser_sha256']:
        raise ValueError('Frozen judge parser changed.')
    inference = inference_files_equal(pilot)
    precision = load(design_dir / 'precision-cost.json')
    selection = load(design_dir / 'selection.json')
    schedule = load(design_dir / 'core-schedule.json')
    exposed = exposed_questions(old_core)
    old_status = load(old_core / 'core-status.json')
    if (precision['exposed_old_core'] != exposed or precision['pilot_protocol_sha256'] != pilot_hash
            or precision['old_core_protocol_sha256'] != digest(old_core / 'protocol.json')
            or precision['quality_scores_used'] is not False or precision['attempts']['total'] != TOTAL
            or precision['selection'] != selection or precision['counts'] != COUNTS
            or selection != selection_for_core(load(pilot / 'selection.json'), exposed)
            or schedule != core_schedule(selection) or len(schedule) != TOTAL
            or not (old_core / 'administrative-stop-accounting.json').exists()
            or old_status['completed'] + 1 != len(list((old_core / 'core-attempts').iterdir()))):
        raise ValueError('The registered design does not reproduce from frozen inputs.')
    verify_dependencies(load(pilot / 'dependencies.json'))
    return {'pilot': {'output': str(pilot), 'protocol_sha256': pilot_hash, 'attempts': 98,
                      'tool_readiness_sha256': digest(pilot / 'tool-readiness.json'),
                      'accounting_sha256': digest(pilot / 'pilot-accounting.json'),
                      'completion_audit_status': pilot_audit['status']},
            'repeatability': {'output': str(repeat), 'protocol_sha256': repeat_hash, 'new_trajectories': 70,
                              'summary_sha256': digest(repeat / 'repeatability-summary.json'),
                              'completion_audit_status': repeat_audit['status']},
            'serving': {'output': str(serving), 'protocol_sha256': serving_hash,
                        'screen_result': 'no client concurrency passes the 10 percent rule',
                        'summary_sha256': digest(serving / 'summary.json')},
            'calibration': {'output': str(calibration), 'protocol_sha256': digest(calibration / 'protocol.json'),
                            'responses': 42, 'judge_requests': cal_summary['judge_requests'],
                            'human_review': 'pending', 'quality_status': 'provisional'},
            'old_core': {'output': str(old_core), 'protocol_sha256': digest(old_core / 'protocol.json'),
                         'completed_attempts': old_status['completed'], 'interrupted_attempts': 1,
                         'exposed_questions': exposed, 'use': 'exposed development material; never combined'},
            'design': {'directory': str(design_dir), 'precision_cost_sha256': digest(design_dir / 'precision-cost.json'),
                       'selection_sha256': digest(design_dir / 'selection.json'),
                       'schedule_sha256': digest(design_dir / 'core-schedule.json'), 'attempts': TOTAL},
            'inference_files_equal_to_v3': inference,
            'scoring_files_equal_to_calibration': {n: cal_protocol['source_sha256'][n] for n in SCORING_FILES},
            'loader_files_equal_to_v3': {n: pilot_manifest['evaluation/agentic_tools/' + n] for n in LOADER_FILES},
            'loader_disclosure': 'records.py gained a provider-journal integrity check after calibration (pilot v2); '
                                 'it changes no score and equals the v3 and core-v1 snapshots.'}


def build_scenarios(pilot, schedule, inputs):
    """Reuse frozen scenarios; add fresh BrowseComp questions from the pinned query source only."""
    existing = load(pilot / 'inference/scenarios.json')
    by_identity = {(v['dataset'], v['id'], v['variant']): v for v in existing.values()}
    data = Path(inputs['browsecomp-plus']['data'])
    manifest = load(data / 'manifest.json')
    if (digest(data / 'manifest.json') != inputs['browsecomp-plus']['data_manifest_sha256']
            or digest(data / 'queries.json') != manifest['files']['queries.json']):
        raise ValueError('BrowseComp query source changed.')
    queries = load(data / 'queries.json')
    if set(queries) != set(manifest['query_ids']):
        raise ValueError('BrowseComp query population changed.')
    scenarios = {}
    for row in schedule:
        identity = (row['dataset'], row['id'], row['variant'])
        sid = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        if sid in scenarios:
            continue
        case = by_identity.get(identity)
        if case is not None:
            if case['scenario_id'] != sid or case['split'] == 'pilot':
                raise ValueError('Exposed pilot scenario cannot enter the core.')
            scenarios[sid] = case
        elif row['dataset'] == 'browsecomp-plus':
            scenarios[sid] = {'scenario_id': sid, 'dataset': 'browsecomp-plus', 'id': row['id'], 'variant': 'v0',
                              'split': 'core', 'query': queries[row['id']]}
        else:
            raise ValueError('Missing prepared scenario: ' + json.dumps(identity))
    return scenarios


def prepare(pilot, old_core, repeat, serving, calibration, design_dir, out, public):
    evidence = gate_evidence(pilot, old_core, repeat, serving, calibration, design_dir)
    tests_passed = check_tests(public / 'tests.xml')
    usage = shutil.disk_usage(out.parent)
    if usage.free < MINIMUM_FREE_GIB * 2 ** 30:
        raise ValueError('Insufficient free space on the output volume.')
    pilot_protocol = load(pilot / 'protocol.json')
    out.mkdir(parents=True, exist_ok=False)
    for name in COPIED_INPUTS:
        if digest(pilot / name) != pilot_protocol['files'][name]:
            raise ValueError('Pilot input changed: ' + name)
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pilot / name, out / name)
    for name in ('selection.json', 'core-schedule.json', 'precision-cost.json'):
        shutil.copyfile(design_dir / name, out / name)
    selection, schedule = load(out / 'selection.json'), load(out / 'core-schedule.json')
    scenarios = build_scenarios(pilot, schedule, load(out / 'inputs.json'))
    if len(scenarios) != COUNTS['browsecomp-plus'] + COUNTS['fiqa'] + COUNTS['nfcorpus'] + 2 * COUNTS['musique']:
        raise ValueError('Unexpected scenario count.')
    (out / 'inference').mkdir()
    write_json(out / 'inference/scenarios.json', scenarios)
    design = load(pilot / 'design.json')
    design.update(status='replacement_core_v2', pilot_attempts=98, core_attempts=TOTAL,
                  primary_attempts=PRIMARY_TOTAL, repeat_subset_attempts=REPEAT_TOTAL,
                  primary_repetitions=list(PRIMARY_REPETITIONS),
                  repeat_subset={'dataset': REPEAT_SUBSET['dataset'], 'ids': repeat_ids(selection),
                                 'repetitions': list(REPEAT_SUBSET['repetitions'])},
                  selection_sha256=digest(out / 'selection.json'),
                  note='Replacement core v2: fresh questions by frozen hash order, repetition zero primary, '
                       'registered 20-question BrowseComp repeat subset; same model, prompts, budgets and tools as pilot v3.')
    write_json(out / 'design.json', design)
    copies = [(ROOT / 'docs/agentic-core-v2-protocol.md', 'core-protocol.md'),
              (pilot / 'tool-readiness.json', 'pilot-tool-readiness.json'),
              (pilot / 'pilot-accounting.json', 'pilot-accounting.json'),
              (ROOT / 'evaluation/agentic-tools/v3/completion-audit.json', 'pilot-completion-audit.json'),
              (repeat / 'repeatability-summary.json', 'repeatability-summary.json'),
              (ROOT / 'evaluation/agentic-tools/repeatability-v1/completion-audit.json', 'repeatability-completion-audit.json'),
              (serving / 'summary.json', 'serving-summary.json'),
              (ROOT / 'evaluation/agentic-tools/serving-v1/completion-audit.json', 'serving-completion-audit.json'),
              (calibration / 'summary.json', 'calibration-summary.json'),
              (calibration / 'protocol.json', 'calibration-protocol.json'),
              (calibration / 'controls.json', 'calibration-controls.json'),
              (public / 'tests.xml', 'tests.xml')]
    for origin, name in copies:
        shutil.copyfile(origin, out / name)
    write_json(out / 'gate-evidence.json', {**evidence, 'prepared_at': utc(), 'tests_passed': tests_passed,
                                             'public_dir': str(public), 'core_dispatch_authorized': False})
    prepared = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
    write_json(out / 'registration-inputs.json', {
        'created_at': utc(), 'working_files': working_files(),
        'prepared_files': {name: digest(out / name) for name in prepared}, 'tests_passed': tests_passed,
        'qualification': 'Pinned before resources were available; registration refuses any change.'})
    write_json(out / 'preparation-status.json', {'status': 'prepared_not_registered', 'updated_at': utc(),
                                                 'attempts': TOTAL, 'scenarios': len(scenarios),
                                                 'core_dispatch_authorized': False})
    return evidence


def register(out, public):
    """Freeze the executable core protocol once resources and identities verify."""
    if (out / 'protocol.json').exists():
        raise ValueError('Protocol already exists; use the registered run.')
    verify_registration_inputs(out)
    evidence = load(out / 'gate-evidence.json')
    pilot = Path(evidence['pilot']['output'])
    if digest(pilot / 'protocol.json') != evidence['pilot']['protocol_sha256']:
        raise ValueError('Pilot protocol changed after preparation.')
    parent = load(pilot / 'protocol.json')
    preflight = load(out / 'provider-preflight.json')
    if preflight['status'] != 'passed' or preflight['chat_model'] != parent['models']['chat']:
        raise ValueError('Provider preflight has not passed with the registered chat model.')
    with httpx.Client(base_url='http://127.0.0.1:11434', timeout=30) as http:
        if http.get('/api/version').raise_for_status().json() != parent['ollama_version']:
            raise ValueError('Model server changed after pilot v3.')
        for identity in parent['models'].values():
            if model_identity(http, identity['name']) != identity:
                raise ValueError('Model identity changed: ' + identity['name'])
    verify_dependencies(load(out / 'dependencies.json'))
    inference = inference_files_equal(pilot)
    if gpu_competitors():
        raise ValueError('Competing GPU work prevents registration.')
    source = out / 'measured-source'
    shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'evaluation/agentic_tools', source / 'evaluation/agentic_tools',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    manifest = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob('*') if p.is_file()}
    write_json(out / 'measured-source-manifest.json', manifest)
    files = sorted(load(out / 'registration-inputs.json')['prepared_files']) + [
        'registration-inputs.json', 'provider-preflight.json', 'measured-source-manifest.json']
    selection = load(out / 'selection.json')
    precision = load(out / 'precision-cost.json')
    protocol = {**parent}
    for key in ('amendment_sha256', 'core_schedule_scope', 'core_gate', 'administrative_stop', 'files',
                'parent_output', 'parent_protocol_sha256', 'revision'):
        protocol.pop(key, None)
    protocol.update(
        schema='agentic-tools-v1-executable-core', phase='core', created_at=utc(), revision=REVISION,
        output=str(out), public_dir=str(public), measured_source=str(source),
        parent_output=str(pilot), parent_protocol_sha256=evidence['pilot']['protocol_sha256'],
        pilot_attempts=98, core_attempts=TOTAL, primary_attempts=PRIMARY_TOTAL, repeat_attempts=REPEAT_TOTAL,
        core_dispatch_authorized=True,
        core_gate='v3 tool readiness passed and audited; serving screen negative; repeatability audited; '
                  'unchanged inference files; frozen scoring identical to calibration; registered outcome-blind '
                  'design; regression tests; provider preflight; model, server and dependency identities; free GPU',
        readiness_policy='readiness-policy.json', core_operational_policy=CORE_OPERATIONAL_POLICY,
        administrative_stop='SIGINT/SIGTERM or stop-request.json; finish the current trajectory, persist the stop; '
                            'review before any explicit resume; never automatically retry',
        statistics={**STATISTICS, 'primary_repetitions': list(PRIMARY_REPETITIONS),
                    'repeat_subset': {'dataset': REPEAT_SUBSET['dataset'], 'ids': repeat_ids(selection),
                                      'repetitions': list(REPEAT_SUBSET['repetitions']),
                                      'estimand': precision['repeat_subset']['estimand']}},
        estimands=precision['estimands'],
        quality_status='provisional_pending_independent_human_review', human_review='pending',
        inference_files_equal_to_v3=inference,
        runner_change='generic Metal-device GPU competitor detection, registered core pause rules and trajectory '
                      'time cap in the harness; no model input, tool, budget or prompt change',
        analysis_change='protocol-driven attempt and grade counts, repetition-zero primary analysis, registered '
                        'repeat-subset variance decomposition',
        cost_estimate={'expected_trajectory_hours': precision['cost']['expected_trajectory_hours'],
                       'planning_hours_with_25_percent_headroom': precision['cost']['planning_hours_with_25_percent_headroom'],
                       'trajectory_time_cap_hours': precision['cost']['trajectory_time_cap_hours'],
                       'excludes': precision['cost']['excluded']},
        exposure={'public_labels': 'all four benchmarks and their labels are public; no private unseen test set is claimed',
                  'development_material': {'pilot_ids': {ds: [i for s in strata for i in s['development_excluded']]
                                                         for ds, strata in selection['tracks'].items()},
                                           'old_core_started': evidence['old_core']['exposed_questions']},
                  'core_v2_questions_inspected_before_inference': 0},
        gates=evidence, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        files={name: digest(out / name) for name in files})
    write_json(out / 'protocol.json', protocol)
    write_json(public / 'protocol.json', protocol)
    return digest(out / 'protocol.json')


def register_if_available(out, public):
    parent_lock = load(out / 'gate-evidence.json')
    pilot_protocol = load(Path(parent_lock['pilot']['output']) / 'protocol.json')
    lock_path = Path(pilot_protocol['inference_lock'])
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'waiting_for_inference_lock'}
        competitors = gpu_competitors()
        if competitors:
            return {'status': 'waiting_for_gpu', 'competing_processes': competitors}
        if (out / 'protocol.json').exists():
            protocol = load(out / 'protocol.json')
            verify_files(out, protocol)
            return {'status': 'registered', 'protocol_sha256': digest(out / 'protocol.json')}
        verify_registration_inputs(out)
        from evaluation.agentic_tools import preflight
        preflight.main(['--output', str(out)])
        competitors = gpu_competitors()
        if competitors:
            return {'status': 'waiting_for_gpu', 'competing_processes': competitors}
        return {'status': 'registered', 'protocol_sha256': register(out, public)}


def accounting(out, *, public=None):
    """Offline accounting of the registered run; complete or partial, never quality-based."""
    protocol, records = read_records(out, require_complete=False)
    policy = load(out / 'readiness-policy.json')
    design = load(out / 'design.json')
    total = protocol['core_attempts']
    review = out / 'operational-review.json'
    extension = out / 'time-cap-extension.json'
    guard = CoreGuard(policy, protocol['core_operational_policy'],
                      reviewed=load(review)['reviewed_attempts'] if review.exists() else [],
                      cap_hours=load(extension)['hours'] if extension.exists() else None)
    groups, reasons, checks = defaultdict(list), [], 0
    for row in records:
        checks += audit_record(row, options=design['options'], think=design['think'])['checks']
        reasons.extend(guard.observe(row))
        s = row['schedule']
        groups[(s['dataset'], s['arm'], s['repetition'])].append(row)
    complete = len(records) == total
    gate = assess(records, policy, complete=complete) if records else None
    metrics = []
    for (dataset, arm, repetition), rows in sorted(groups.items()):
        times = [r['elapsed_ms'] / 1000 for r in rows]
        statuses = Counter((r.get('result') or {}).get('final', {}).get('status', 'error') for r in rows)
        stops = Counter((r.get('result') or {}).get('observation', {}).get('budget_stop_reason') or
                        (r.get('result') or {}).get('stop_reason', 'error') for r in rows)
        metrics.append({'dataset': dataset, 'arm': arm, 'repetition': repetition, 'attempts': len(rows),
                        'seconds_mean': float(np.mean(times)), 'seconds_p50': float(np.median(times)),
                        'seconds_p95': float(np.quantile(times, .95)), 'final_statuses': dict(statuses),
                        'stop_reasons': dict(stops), 'harness_errors': sum(bool(r.get('error')) for r in rows),
                        'timing_contamination_flags': sum(bool(r.get('competing_processes_after')) for r in rows),
                        'model_requests': sum(len((r.get('result') or {}).get('observation', {}).get('models', [])) for r in rows)})
    setup_path = out / 'core-setup-costs.json'
    setup = load(setup_path) if setup_path.exists() else []
    manifest_path = out / 'scope-readiness-manifest.json'
    fixtures = load(manifest_path) if manifest_path.exists() else {}
    summary = {'schema': 'agentic-core-v2-accounting', 'status': 'complete' if complete else 'partial',
               'checked_at': utc(), 'protocol_sha256': digest(out / 'protocol.json'), 'analysis_sha256': digest(__file__),
               'attempts_accounted': len(records), 'attempts_required': total,
               'primary_attempts_accounted': sum(r['schedule']['repetition'] in PRIMARY_REPETITIONS for r in records),
               'repeat_attempts_accounted': sum(r['schedule']['repetition'] not in PRIMARY_REPETITIONS for r in records),
               'trace_checks': checks, 'groups': metrics,
               'dataset_seconds': {ds: sum(r['elapsed_ms'] / 1000 for r in records if r['schedule']['dataset'] == ds)
                                   for ds in COUNTS},
               'measured_hours': guard.measured_seconds / 3600, 'guard': guard.state(),
               'pause_reasons_on_replay': reasons,
               'tool_groups': gate['groups'] if gate else [], 'execution_failures': gate['execution_failures'] if gate else [],
               'operational_failures': sum(g['unexpected_errors'] for g in gate['groups']) if gate else 0,
               'failure_categories': dict(Counter(c for r in records if (c := failure_category(r)))),
               'setup_seconds': sum(s['elapsed_ms'] for s in setup) / 1000, 'scope_preparations': len(setup),
               'retained_scope_checks': len(fixtures),
               'quality_scores_used': False, 'answer_quality': 'not scored here; grading and independent review are separate'}
    write_json(out / 'core-accounting.json', summary)
    if public:
        write_json(public / 'core-accounting.json', summary)
    return summary


def run_stage(argv, *, cwd, env=None):
    return subprocess.run([sys.executable, '-u', '-m', *argv], cwd=cwd, env=env).returncode


def post_inference(out, public, source, env, status, stop, wait_for_gpu):
    status(status='accounting')
    summary = accounting(out, public=public)
    if summary['status'] != 'complete':
        raise RuntimeError('Accounting is incomplete after a completed worker.')
    status(status='auditing')
    audit_env = {**os.environ, 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'}
    if run_stage(['evaluation.audits.audit_agentic_core_v2', '--output', str(out), '--report', str(public / 'completion-audit.json')],
                 cwd=ROOT, env=audit_env):
        raise RuntimeError('Independent core audit failed.')
    grading = out / 'grading'
    while not (grading / 'summary.json').exists():
        if stop.requested():
            status(status='paused', stage='before_grading', administrative_stop=stop.persist())
            return
        competitors = gpu_competitors()
        if competitors:
            status(status='waiting_for_gpu', stage='before_grading', competing_processes=competitors)
            if not wait_for_gpu:
                return
            time.sleep(30)
            continue
        status(status='grading')
        code = run_stage(['evaluation.agentic_tools.judge', '--source', str(out), '--output', str(grading)], cwd=source, env=env)
        if code and not (grading / 'summary.json').exists():
            state = load(grading / 'status.json') if (grading / 'status.json').exists() else {}
            message = str((state.get('error') or {}).get('message', ''))
            if 'Competing GPU' in message and wait_for_gpu:
                time.sleep(30)
                continue
            raise RuntimeError('Judge failed: ' + message)
    status(status='analyzing')
    if run_stage(['evaluation.agentic_tools.analyze', '--output', str(out), '--grading', str(grading),
                  '--destination', str(out / 'analysis')], cwd=ROOT, env=audit_env):
        raise RuntimeError('Analysis failed.')
    for name in ('summary.json', 'citation-review-pending.json', 'analysis-checksums.json'):
        shutil.copyfile(out / 'analysis' / name, public / ('analysis-' + name))
    shutil.copyfile(grading / 'summary.json', public / 'grading-summary.json')
    status(status='core_complete_analysis_provisional', quality_status='provisional_pending_independent_human_review',
           next_step='English report with Chinese summary; human calibration and citation review remain pending')


def pipeline(out, public, wait_for_gpu):
    meta = {'pid': os.getpid(), 'started_at': utc(), 'output': str(out), 'revision': REVISION}
    def status(**values):
        meta.update(values, updated_at=utc())
        write_json(out / 'pipeline-status.json', meta)
        print(json.dumps(meta), flush=True)
    with (out / 'pipeline.lock').open('a') as job_lock, StopController(out / 'stop-request.json') as stop:
        fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        try:
            while True:
                if stop.requested():
                    status(status='paused', administrative_stop=stop.persist())
                    return
                state = register_if_available(out, public)
                if state != previous:
                    status(**state)
                    previous = state
                else:
                    meta['last_resource_check_at'] = utc()
                    write_json(out / 'pipeline-status.json', meta)
                if state['status'] != 'registered':
                    if not wait_for_gpu:
                        return
                    time.sleep(30)
                    continue
                source = out / 'measured-source'
                env = {**os.environ, 'PYTHONPATH': str(source / 'src') + ':' + str(source),
                       'PYTHONDONTWRITEBYTECODE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
                result = load(out / 'core-status.json') if (out / 'core-status.json').exists() else None
                if not result or result['status'] != 'completed':
                    status(status='running_core')
                    worker = subprocess.Popen([sys.executable, '-u', '-m', 'evaluation.agentic_tools.runner', '--output', str(out)],
                                              env=env, cwd=source)
                    status(worker_pid=worker.pid)
                    while worker.poll() is None:
                        if stop.requested():
                            stop.persist()
                        time.sleep(1)
                    if worker.returncode:
                        raise RuntimeError('Registered core worker failed with exit code ' + str(worker.returncode))
                    result = load(out / 'core-status.json')
                    if result['status'] == 'waiting_for_gpu' and wait_for_gpu:
                        previous = None
                        continue
                    if result['status'] == 'paused':
                        accounting(out, public=public)
                        status(status='paused', completed=result['completed'], total=result['total'],
                               administrative_stop=result.get('administrative_stop'),
                               next_step='review the recorded cause; archive stop-request.json explicitly before any resume')
                        return
                    if result['status'] != 'completed':
                        raise RuntimeError('Core worker ended in state ' + result['status'])
                status(completed=result['completed'], total=result['total'])
                post_inference(out, public, source, env, status, stop, wait_for_gpu)
                return
        except BaseException as exc:
            status(status='failed', error={'type': type(exc).__name__, 'message': str(exc)})
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--public-dir', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--accounting', action='store_true')
    p.add_argument('--pilot', type=Path, default=Path('/Volumes/ARKBPhaseC/agentic-tools-v1/pilot-v3'))
    p.add_argument('--old-core', type=Path, default=Path('/Volumes/ARKBPhaseC/agentic-tools-v1/core-v1'))
    p.add_argument('--repeat', type=Path, default=Path('/Volumes/ARKBPhaseC/agentic-tools-v1/repeatability-v1'))
    p.add_argument('--serving', type=Path, default=Path('/Volumes/ARKBPhaseC/agentic-tools-v1/serving-v1'))
    p.add_argument('--calibration', type=Path, default=Path('/Volumes/ARKBPhaseC/agentic-tools-v1/calibration-v1'))
    p.add_argument('--design', type=Path, default=ROOT / 'evaluation/agentic-tools/core-v2')
    p.add_argument('--wait-for-gpu', action='store_true')
    a = p.parse_args()
    out, public = a.output.resolve(), a.public_dir.resolve()
    if a.prepare:
        evidence = prepare(a.pilot.resolve(), a.old_core.resolve(), a.repeat.resolve(), a.serving.resolve(),
                           a.calibration.resolve(), a.design.resolve(), out, public)
        print(json.dumps({'status': 'prepared_not_registered', 'attempts': TOTAL,
                          'pilot_protocol_sha256': evidence['pilot']['protocol_sha256']}))
    elif a.accounting:
        summary = accounting(out, public=public)
        print(json.dumps({k: summary[k] for k in ('status', 'attempts_accounted', 'attempts_required', 'trace_checks',
                                                   'measured_hours', 'operational_failures', 'pause_reasons_on_replay')}))
    else:
        pipeline(out, public, a.wait_for_gpu)


if __name__ == '__main__':
    main()

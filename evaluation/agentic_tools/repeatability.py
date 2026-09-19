"""Register and analyze bounded full-Agent repeats without answer-based selection."""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from arkb.evaluation.external import digest, write_json
from .contract import ARMS
from .prepare import ROOT, utc
from .records import read_records
from .runner import execute, verify_files
from .summarize_pilot import summarize


def selected_baseline(rows):
    first = {}
    for row in rows:
        s = row['schedule']
        first.setdefault(s['dataset'], s['id'])
    chosen = [r for r in rows if r['schedule']['id'] == first[r['schedule']['dataset']]]
    if len(chosen) != 35:
        raise ValueError('Expected five scenarios and all seven arms.')
    return chosen


def repeat_schedule(baseline):
    units = {}
    for row in baseline:
        s = row['schedule']
        units.setdefault((s['dataset'], s['id']), s)
    result = []
    for (dataset, identifier), base in units.items():
        for repetition in (1, 2):
            offset = (base['unit_index'] + repetition) % len(ARMS)
            for arm in ARMS[offset:] + ARMS[:offset]:
                for variant in (('v0', 'v1') if dataset == 'musique' else ('v0',)):
                    result.append({**base, 'dataset': dataset, 'id': identifier,
                                   'arm': arm.id, 'variant': variant, 'repetition': repetition})
    return result


def prepare(parent, serving, out, public):
    p, rows = read_records(parent)
    verify_files(parent, p)
    if json.loads((parent / 'tool-readiness.json').read_text())['status'] != 'passed':
        raise ValueError('V3 readiness required.')
    screen = json.loads((serving / 'summary.json').read_text())
    if screen['status'] != 'completed' or any(g['eligible_for_agent_confirmation'] for g in screen['groups']):
        raise ValueError('This single-concurrency design requires the completed negative client-concurrency screen.')
    manifest = json.loads((parent / 'measured-source-manifest.json').read_text())
    names = [n for n in manifest if n.startswith('src/')] + [
        'evaluation/agentic_tools/' + n for n in ('contract.py', 'transport.py', 'dependencies.py',
                                                'selection.py', 'readiness.py', 'stopping.py')]
    for name in names:
        if digest(ROOT / name) != manifest[name]:
            raise ValueError('Measured inference behavior changed: ' + name)
    old_runner = (Path(p['measured_source']) / 'evaluation/agentic_tools/runner.py').read_text()
    expected_runner = old_runner.replace("str(pid) + '.json')", "str(pid) + '-' + str(len(setup)) + '.json')")
    if (ROOT / 'evaluation/agentic_tools/runner.py').read_text() != expected_runner:
        raise ValueError('Runner changes exceed the registered auxiliary archive repair.')
    baseline = selected_baseline(rows)
    schedule = repeat_schedule(baseline)
    if len(schedule) != 70:
        raise ValueError('Repeated workload changed.')
    out.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=True)
    for name in ('inputs.json', 'context-indexes.json', 'dependencies.json', 'readiness-policy.json',
                 'provider-preflight.json'):
        shutil.copyfile(parent / name, out / name)
    shutil.copyfile(ROOT / 'docs/agentic-repeatability-v1.md', out / 'plan.md')
    shutil.copyfile(public / 'tests.xml', out / 'tests.xml')
    scenarios = json.loads((parent / 'inference/scenarios.json').read_text())
    ids = {r['scenario_id'] for r in baseline}
    (out / 'inference').mkdir()
    write_json(out / 'inference/scenarios.json', {k: v for k, v in scenarios.items() if k in ids})
    write_json(out / 'pilot-schedule.json', schedule)
    write_json(out / 'core-schedule.json', [])
    design = json.loads((parent / 'design.json').read_text())
    design.update(status='full_agent_development_repeatability', pilot_attempts=70, core_attempts=0)
    write_json(out / 'design.json', design)
    write_json(out / 'baseline-attempts.json', [{
        'key': r['key'], 'schedule': r['schedule'],
        'result_sha256': digest(parent / 'pilot-attempts' / r['key'] / 'result.json'),
        'provider_sha256': digest(parent / 'pilot-attempts' / r['key'] / 'provider.jsonl')} for r in baseline])
    write_json(out / 'serving-evidence.json', {'protocol_sha256': digest(serving / 'protocol.json'),
                                             'summary_sha256': digest(serving / 'summary.json'), 'summary': screen})
    source = out / 'measured-source'
    shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'evaluation/agentic_tools', source / 'evaluation/agentic_tools',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    write_json(out / 'measured-source-manifest.json', {str(f.relative_to(source)): digest(f) for f in source.rglob('*') if f.is_file()})
    inputs = ['inputs.json', 'context-indexes.json', 'dependencies.json', 'readiness-policy.json',
              'provider-preflight.json', 'plan.md', 'tests.xml', 'inference/scenarios.json', 'pilot-schedule.json',
              'core-schedule.json', 'design.json', 'baseline-attempts.json', 'serving-evidence.json', 'measured-source-manifest.json']
    protocol = {**p, 'created_at': utc(), 'revision': 'full-agent-repeatability-v1', 'output': str(out),
                'public_dir': str(public), 'measured_source': str(source), 'parent_output': str(parent),
                'parent_protocol_sha256': digest(parent / 'protocol.json'),
                'pilot_attempts': 70, 'baseline_attempts': 35, 'core_attempts': 0, 'project_core_costs': False,
                'core_dispatch_authorized': False, 'inference_concurrency': 1,
                'core_gate': 'complete repeatability audit and separately justified replacement core protocol',
                'inference_files_equal_to_v3': {n: manifest[n] for n in names},
                'runner_change': 'unique auxiliary scope-check filenames only',
                'development_plan_sha256': digest(out / 'plan.md'),
                'core_schedule_scope': 'none; this development repetition study authorizes no core schedule',
                'files': {n: digest(out / n) for n in inputs}}
    protocol.pop('amendment_sha256', None)
    write_json(out / 'protocol.json', protocol)
    write_json(public / 'protocol.json', protocol)


def fingerprint(row):
    result = row.get('result') or {}
    report = result.get('observation') or {}
    mapping = {ref: 'evidence:' + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
               for ref, value in report.get('evidence_references', {}).items()}
    def normalize(value):
        if isinstance(value, str):
            for ref in sorted(mapping, key=len, reverse=True):
                stable = mapping[ref]
                value = value.replace(ref, stable)
            return value
        if isinstance(value, list): return [normalize(v) for v in value]
        if isinstance(value, dict): return {k: normalize(v) for k, v in value.items()}
        return value
    def hashed(value):
        return hashlib.sha256(json.dumps(normalize(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    tools = [{'name': t['name'], 'arguments': t['arguments'], 'status': t['status']}
             for t in report.get('tools', []) if t['executed']]
    evidence = {'returned': [], 'delivered': [], 'submitted': []}
    for event in report.get('tools', []):
        raw = event.get('raw_result') or {}
        hits = [raw['result']] if 'result' in raw else raw.get('results', [])
        for hit in hits:
            ref = hit.get('ref')
            if ref not in mapping:
                continue
            for stage, include in [('returned', True), ('delivered', event.get('delivered_to_conversation')),
                                   ('submitted', event.get('submitted_to_model'))]:
                if include and mapping[ref] not in evidence[stage]:
                    evidence[stage].append(mapping[ref])
    return {'final_sha256': hashed(result.get('final')), 'tool_sequence_sha256': hashed(tools),
            'evidence_sources_sha256': hashed(row.get('evidence_sources')),
            'evidence_identity_sha256': hashed(evidence),
            'model_requests': len(report.get('models', [])), 'stop_reason': result.get('stop_reason'),
            'final_status': (result.get('final') or {}).get('status', 'error'), 'elapsed_ms': row['elapsed_ms']}


def analyze(out):
    protocol, repeats = read_records(out)
    parent = Path(protocol['parent_output'])
    if digest(parent / 'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Historical baseline protocol changed.')
    _, rows = read_records(parent)
    selected = {r['key']: r for r in selected_baseline(rows)}
    for base in json.loads((out / 'baseline-attempts.json').read_text()):
        path = parent / 'pilot-attempts' / base['key']
        if digest(path / 'result.json') != base['result_sha256'] or digest(path / 'provider.jsonl') != base['provider_sha256']:
            raise ValueError('Retained baseline changed.')
    groups = defaultdict(dict)
    for row in [*selected.values(), *repeats]:
        s = row['schedule']
        groups[s['dataset'], s['id'], s['variant'], s['arm']][s['repetition']] = fingerprint(row)
    cells = []
    for (dataset, identifier, variant, arm), values in sorted(groups.items()):
        if set(values) != {0, 1, 2}:
            raise ValueError('Missing or extra registered repetition.')
        comparisons = {key: {'distinct_all_three': len({v[key] for v in values.values()}),
                              'new_repeats_equal': values[1][key] == values[2][key]}
                       for key in ('final_sha256', 'tool_sequence_sha256', 'evidence_sources_sha256', 'evidence_identity_sha256',
                                   'model_requests', 'stop_reason', 'final_status')}
        cells.append({'dataset': dataset, 'id': identifier, 'variant': variant, 'arm': arm,
                      'executions': values, 'comparisons': comparisons})
    result = {'status': 'completed', 'completed_at': utc(), 'protocol_sha256': digest(out / 'protocol.json'),
              'new_trajectories': len(repeats), 'historical_baseline_trajectories': len(selected), 'cells': cells,
              'independent_questions': 4, 'quality_scores_used': False, 'core_dispatch_authorized': False}
    write_json(out / 'repeatability-summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--parent', type=Path)
    p.add_argument('--serving', type=Path)
    p.add_argument('--public-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if a.prepare:
        prepare(a.parent.resolve(), a.serving.resolve(), out, a.public_dir.resolve())
    else:
        execute(out)
        status = json.loads((out / 'pilot-status.json').read_text())
        summarize(out)
        if status['status'] == 'completed':
            result = analyze(out)
            print(json.dumps({k: result[k] for k in ('status', 'new_trajectories', 'historical_baseline_trajectories')}), flush=True)


if __name__ == '__main__':
    main()

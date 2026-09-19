"""Audit registered full-Agent repeats from immutable attempts and HTTP journals."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from ollama import ChatResponse

from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.prepare import utc
from evaluation.agentic_tools.readiness import assess
from evaluation.agentic_tools.records import read_records
from evaluation.agentic_tools.runner import verify_files
from evaluation.agentic_tools.summarize_pilot import audit_record


def fingerprint(row):
    """Independent reconstruction; substitute complete known tokens in one pass."""
    report = row['result']['observation']
    def checksum(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    refs = {key: 'evidence:' + checksum(value) for key, value in report['evidence_references'].items()}
    pattern = re.compile('|'.join(re.escape(key) for key in sorted(refs, key=len, reverse=True))) if refs else None
    def normalize(value):
        if isinstance(value, str):
            return pattern.sub(lambda m: refs[m.group()], value) if pattern else value
        if isinstance(value, list):
            return [normalize(v) for v in value]
        if isinstance(value, dict):
            return {key: normalize(v) for key, v in value.items()}
        return value
    tools = [{key: event[key] for key in ('name', 'arguments', 'status')}
             for event in report['tools'] if event['executed']]
    stages = {stage: [] for stage in ('returned', 'delivered', 'submitted')}
    for event in report['tools']:
        raw = event.get('raw_result') or {}
        hits = [raw['result']] if 'result' in raw else raw.get('results', [])
        for hit in hits:
            ref = refs.get(hit.get('ref'))
            if ref is None:
                continue
            for stage, visible in [('returned', True), ('delivered', event.get('delivered_to_conversation')),
                                   ('submitted', event.get('submitted_to_model'))]:
                if visible and ref not in stages[stage]:
                    stages[stage].append(ref)
    return {'final_sha256': checksum(normalize(row['result'].get('final'))),
            'tool_sequence_sha256': checksum(normalize(tools)),
            'evidence_sources_sha256': checksum(normalize(row['evidence_sources'])),
            'evidence_identity_sha256': checksum(stages), 'model_requests': len(report['models']),
            'stop_reason': row['result'].get('stop_reason'),
            'final_status': (row['result'].get('final') or {}).get('status', 'error'),
            'elapsed_ms': row['elapsed_ms']}


def audit(out, destination):
    protocol, repeats = read_records(out)
    verify_files(out, protocol)
    parent = Path(protocol['parent_output'])
    if digest(parent / 'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed.')
    parent_protocol, parent_rows = read_records(parent)
    verify_files(parent, parent_protocol)
    parent_by_key = {r['key']: r for r in parent_rows}
    baseline_manifest = json.loads((out / 'baseline-attempts.json').read_text())
    baseline = [parent_by_key[b['key']] for b in baseline_manifest]
    first = {}
    for row in parent_rows:
        first.setdefault(row['schedule']['dataset'], row['schedule']['id'])
    expected_baseline = [r['key'] for r in parent_rows if r['schedule']['id'] == first[r['schedule']['dataset']]]
    if expected_baseline != [b['key'] for b in baseline_manifest] or len(baseline) != 35 or len(repeats) != 70:
        raise ValueError('Outcome-independent cohort or counts changed.')
    checksums, request_count, trace_checks = {}, Counter(), Counter()
    for root, rows, registration, label in [(parent, baseline, parent_protocol, 'baseline'),
                                            (out, repeats, protocol, 'new')]:
        design = json.loads((root / 'design.json').read_text())
        for row in rows:
            key = row['key']
            path = root / 'pilot-attempts' / key
            manifest = json.loads((path / 'complete.json').read_text())
            checksums[label + ':' + key] = manifest
            if label == 'baseline':
                pinned = next(b for b in baseline_manifest if b['key'] == key)
                if any(pinned[field] != manifest[field] for field in ('result_sha256', 'provider_sha256')):
                    raise ValueError('Pinned historical attempt changed.')
            attempt = json.loads((path / 'attempt.json').read_text())
            if any(attempt[field] != row[field] for field in ('key', 'protocol_sha256', 'schedule')):
                raise ValueError('Attempt start record differs from completed record.')
            trace_checks[label] += audit_record(row, options=design['options'], think=design['think'])['checks']
            events = [json.loads(line) for line in (path / 'provider.jsonl').read_text().splitlines()]
            models = row['result']['observation']['models']
            if len(events) != 2 * len(models) or row['error'] or row['competing_processes_after']:
                raise ValueError('Missing HTTP events, harness failure or resource competition.')
            for index, model in enumerate(models):
                request, response = events[2 * index:2 * index + 2]
                if (request['event'] != 'provider_request' or request['request'] != model['request']
                        or response['event'] != 'provider_response' or response['http_status'] != 200):
                    raise ValueError('Actual HTTP journal differs from observed model call.')
                actual = ChatResponse.model_validate(json.loads(response['body'])).model_dump(exclude_none=True)
                if actual != model['response'] or not actual['done'] or actual['message'].get('thinking'):
                    raise ValueError('Recorded completion differs from raw provider response.')
                request_count[label] += 1
            loaded = next(m for m in row['loaded_models_after']['models'] if m['name'] == registration['models']['chat']['name'])
            if loaded['digest'] != registration['models']['chat']['digest'] or loaded['context_length'] != design['options']['num_ctx']:
                raise ValueError('Effective model or context changed.')
        if label == 'new' and {p.name for p in (root / 'pilot-attempts').iterdir()} != {r['key'] for r in rows}:
            raise ValueError('Unexpected attempt directory.')

    policy = json.loads((out / 'readiness-policy.json').read_text())
    gate = assess(repeats, policy, complete=True)
    saved_gate = json.loads((out / 'tool-readiness.json').read_text())
    if any(saved_gate[key] != value for key, value in gate.items()) or gate['status'] != 'passed':
        raise ValueError('Tool readiness differs on replay.')
    setup = json.loads((out / 'pilot-setup-costs.json').read_text())
    fixture_manifest = json.loads((out / 'scope-readiness-manifest.json').read_text())
    fixture_calls = 0
    if len(setup) != 31 or len(fixture_manifest) != len(setup):
        raise ValueError('A scope preparation lost its auxiliary check.')
    expected_names = set()
    for index, prepared in enumerate(setup, 1):
        name = hashlib.sha256(prepared['scope'].encode()).hexdigest() + f"-{prepared['pid']}-{index}.json"
        expected_names.add(name)
        path = out / 'scope-readiness' / name
        if fixture_manifest[name] != digest(path):
            raise ValueError('Scope fixture changed.')
        fixture = json.loads(path.read_text())
        if (fixture['protocol_sha256'] != digest(out / 'protocol.json') or fixture['scope'] != prepared['scope']
                or fixture['index_id'] != prepared['index_id'] or fixture['status'] != 'passed'
                or len(fixture['calls']) != len(policy['workload'])):
            raise ValueError('Scope fixture identity or workload changed.')
        for call, planned in zip(fixture['calls'], policy['workload'], strict=True):
            if call['status'] != 'success' or call['error'] or call['seconds'] >= policy['timely_seconds']:
                raise ValueError('A scope fixture failed.')
            if planned['name'] != 'read' and call['arguments'] != planned['arguments']:
                raise ValueError('Scope fixture arguments differ from registration.')
            fixture_calls += 1
    if set(fixture_manifest) != expected_names or {p.name for p in (out / 'scope-readiness').iterdir()} != expected_names:
        raise ValueError('Unexpected auxiliary check artifact.')

    cells, action_cells = defaultdict(dict), defaultdict(dict)
    for row in baseline + repeats:
        s = row['schedule']
        cell = cells[s['dataset'], s['id'], s['variant'], s['arm']]
        if str(s['repetition']) in cell:
            raise ValueError('Duplicate repetition.')
        cell[str(s['repetition'])] = fingerprint(row)
        actions = deepcopy(row)
        actions['result']['observation']['tools'] = [t for t in actions['result']['observation']['tools'] if t['name'] != 'finish']
        action_cells[s['dataset'], s['id'], s['variant'], s['arm']][s['repetition']] = fingerprint(actions)['tool_sequence_sha256']
    summary = json.loads((out / 'repeatability-summary.json').read_text())
    if len(summary['cells']) != 35 or len(cells) != 35:
        raise ValueError('Missing scenario/arm cell.')
    aggregate = defaultdict(Counter)
    for saved in summary['cells']:
        values = cells[saved['dataset'], saved['id'], saved['variant'], saved['arm']]
        if values != saved['executions'] or set(values) != {'0', '1', '2'}:
            raise ValueError('Repeated fingerprints differ on independent reconstruction.')
        for field, comparison in saved['comparisons'].items():
            distinct = len({v[field] for v in values.values()})
            equal = values['1'][field] == values['2'][field]
            if comparison != {'distinct_all_three': distinct, 'new_repeats_equal': equal}:
                raise ValueError('Repeatability comparisons differ on replay.')
            aggregate[field]['all_three_equal'] += distinct == 1
            aggregate[field]['new_two_equal'] += equal
    status = json.loads((out / 'pilot-status.json').read_text())
    if status['status'] != 'completed' or status['completed'] != 70:
        raise ValueError('Runner did not complete.')
    seconds = (datetime.fromisoformat(status['updated_at']) - datetime.fromisoformat(status['started_at'])).total_seconds()
    report = {'status': 'passed', 'checked_at': utc(), 'audit_source_sha256': digest(__file__),
              'protocol_sha256': digest(out / 'protocol.json'), 'new_trajectories': 70,
              'baseline_trajectories': 35, 'independent_questions': 4, 'scenario_arm_cells': 35,
              'provider_requests': dict(request_count), 'trace_checks': dict(trace_checks),
              'tools': {'calls': sum(g['calls'] for g in gate['groups']),
                        'valid_calls': sum(g['valid_calls'] for g in gate['groups']),
                        'input_errors': sum(g['input_errors'] for g in gate['groups']),
                        'operational_errors': sum(g['unexpected_errors'] for g in gate['groups'])},
              'scope_preparations': len(setup), 'retained_scope_checks': len(fixture_manifest),
              'scope_fixture_calls': fixture_calls, 'stability': dict(aggregate),
              'supplementary_actions_excluding_finish': {
                  'all_cells': len(action_cells),
                  'all_three_equal': sum(len(set(v.values())) == 1 for v in action_cells.values()),
                  'new_two_equal': sum(v[1] == v[2] for v in action_cells.values()),
                  'agent_cells': sum(k[-1].startswith('A-') for k in action_cells),
                  'agent_new_two_equal': sum(v[1] == v[2] for k, v in action_cells.items() if k[-1].startswith('A-')),
                  'scope': 'Post-run explanatory diagnostic; registered tool fingerprint includes finish.'},
              'new_trajectory_seconds': sum(r['elapsed_ms'] for r in repeats) / 1000,
              'setup_seconds': sum(s['elapsed_ms'] for s in setup) / 1000,
              'worker_wall_seconds': seconds, 'attempt_checksums': checksums,
              'artifact_checksums': {n: digest(out / n) for n in ('pilot-accounting.json', 'tool-readiness.json',
                 'repeatability-summary.json', 'pilot-status.json', 'pilot-setup-costs.json', 'scope-readiness-manifest.json')},
              'quality_scored': False, 'population_repeatability_established': False,
              'core_dispatch_authorized': False}
    write_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output, args.report)
    print(json.dumps({k: v for k, v in result.items() if k not in ('attempt_checksums', 'artifact_checksums')}, indent=2))


if __name__ == '__main__':
    main()

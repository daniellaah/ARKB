"""Independent audit of the registered core v2 run from immutable attempts and HTTP journals.

Verifies identities, outcome-blind selection and scheduling, scenario provenance,
provider journals, effective model settings, scope fixtures and the registered
pause rules. It reads no answer labels and produces no quality claim.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

from ollama import ChatResponse

from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.contract import ARMS
from evaluation.agentic_tools.core_design import (COUNTS, PRIMARY_REPETITIONS, REPEAT_SUBSET, TOTAL, core_schedule,
                                                  exposed_questions, repeat_ids, selection_for_core)
from evaluation.agentic_tools.prepare import utc
from evaluation.agentic_tools.readiness import assess
from evaluation.agentic_tools.records import read_records
from evaluation.agentic_tools.runner import CoreGuard, verify_files
from evaluation.agentic_tools.summarize_pilot import audit_record


def load(path):
    return json.loads(Path(path).read_text())


def audit(out, destination):
    protocol, rows = read_records(out)
    verify_files(out, protocol)
    protocol_hash = digest(out / 'protocol.json')
    if (protocol['phase'] != 'core' or protocol['revision'] != 'replacement-core-v2'
            or protocol['core_dispatch_authorized'] is not True or protocol['core_attempts'] != TOTAL or len(rows) != TOTAL):
        raise ValueError('Not a complete registered core v2 run.')
    pilot = Path(protocol['parent_output'])
    if digest(pilot / 'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent pilot protocol changed.')
    old_core = Path(protocol['gates']['old_core']['output'])
    exposed = exposed_questions(old_core)
    selection = load(out / 'selection.json')
    schedule = load(out / 'core-schedule.json')
    if selection != selection_for_core(load(pilot / 'selection.json'), exposed) or schedule != core_schedule(selection):
        raise ValueError('Outcome-independent selection or schedule does not reproduce.')
    pilot_selection = load(pilot / 'selection.json')
    for dataset, strata in pilot_selection['tracks'].items():
        for pilot_stratum, core_stratum in zip(strata, selection['tracks'][dataset], strict=True):
            banned = set(pilot_stratum['pilot']) | set(exposed.get(dataset, []))
            if banned & set(core_stratum['core']):
                raise ValueError('Exposed development question inside the core.')
    repeated = set(repeat_ids(selection))
    if protocol['statistics']['repeat_subset']['ids'] != repeat_ids(selection):
        raise ValueError('Registered repeat subset differs from the schedule prefix rule.')
    counts = Counter((r['schedule']['dataset'], r['schedule']['arm'], r['schedule']['repetition']) for r in rows)
    for arm in ARMS:
        for dataset, count in COUNTS.items():
            expected = 2 * count if dataset == 'musique' else count
            if counts[(dataset, arm.id, 0)] != expected:
                raise ValueError('Primary attempt count differs from registration.')
        for repetition in REPEAT_SUBSET['repetitions'][1:]:
            if counts[('browsecomp-plus', arm.id, repetition)] != len(repeated):
                raise ValueError('Repeat subset count differs from registration.')
    scenarios = load(out / 'inference/scenarios.json')
    pilot_scenarios = load(pilot / 'inference/scenarios.json')
    inputs = load(out / 'inputs.json')
    data = Path(inputs['browsecomp-plus']['data'])
    manifest = load(data / 'manifest.json')
    if digest(data / 'manifest.json') != inputs['browsecomp-plus']['data_manifest_sha256'] or digest(data / 'queries.json') != manifest['files']['queries.json']:
        raise ValueError('BrowseComp query source changed.')
    queries = load(data / 'queries.json')
    for sid, case in scenarios.items():
        identity = (case['dataset'], case['id'], case['variant'])
        if hashlib.sha256(json.dumps(identity).encode()).hexdigest() != sid or case['split'] == 'pilot':
            raise ValueError('Scenario identity or split is wrong.')
        original = pilot_scenarios.get(sid)
        if original is not None and original != case:
            raise ValueError('A reused scenario differs from the frozen pilot scenario.')
        if original is None and (case['dataset'] != 'browsecomp-plus' or case['query'] != queries[case['id']]):
            raise ValueError('A fresh scenario is not the pinned official query text.')
    design = load(out / 'design.json')
    policy = load(out / 'readiness-policy.json')
    review = out / 'operational-review.json'
    extension = out / 'time-cap-extension.json'
    reviewed = load(review)['reviewed_attempts'] if review.exists() else []
    guard = CoreGuard(policy, protocol['core_operational_policy'], reviewed=reviewed,
                      cap_hours=load(extension)['hours'] if extension.exists() else None)
    checksums, request_count, trace_checks, reasons = {}, 0, 0, []
    harness_errors, contaminated = [], []
    directories = {p.name for p in (out / 'core-attempts').iterdir()}
    if directories != {r['key'] for r in rows}:
        raise ValueError('Unexpected or missing attempt directory.')
    for row in rows:
        key = row['key']
        path = out / 'core-attempts' / key
        manifest_entry = load(path / 'complete.json')
        checksums[key] = manifest_entry
        attempt = load(path / 'attempt.json')
        if any(attempt[field] != row[field] for field in ('key', 'protocol_sha256', 'schedule')):
            raise ValueError('Attempt start record differs from completed record.')
        trace_checks += audit_record(row, options=design['options'], think=design['think'])['checks']
        events = [json.loads(line) for line in (path / 'provider.jsonl').read_text().splitlines()]
        report = (row.get('result') or {}).get('observation')
        if row.get('error'):
            harness_errors.append(key)
        if row.get('competing_processes_after'):
            contaminated.append(key)
        if report:
            models = report['models']
            if len(events) < 2 * len(models):
                raise ValueError('Missing HTTP events for observed model calls.')
            for index, model in enumerate(models):
                request, response = events[2 * index:2 * index + 2]
                if request['event'] != 'provider_request' or request['request'] != model['request'] or response['event'] != 'provider_response':
                    raise ValueError('Actual HTTP journal differs from observed model call.')
                if response['http_status'] == 200 and model.get('response'):
                    actual = ChatResponse.model_validate(json.loads(response['body'])).model_dump(exclude_none=True)
                    if actual != model['response'] or actual['message'].get('thinking'):
                        raise ValueError('Recorded completion differs from raw provider response.')
                request_count += 1
        loaded = [m for m in row['loaded_models_after']['models'] if m['name'] == protocol['models']['chat']['name']]
        if loaded and (loaded[0]['digest'] != protocol['models']['chat']['digest'] or loaded[0]['context_length'] != design['options']['num_ctx']):
            raise ValueError('Effective model or context changed.')
        reasons.extend(guard.observe(row))
    keyed = [r for r in reasons if ':' in r and not r.startswith('operational_failure_rate')]
    unreviewed = [r for r in keyed if r.split(':', 1)[1] not in set(reviewed)]
    if unreviewed:
        raise ValueError('A registered pause rule fired on an attempt that no operational review acknowledges: ' + unreviewed[0])
    if any(r == 'trajectory_time_cap_reached' for r in reasons) and not extension.exists():
        raise ValueError('The trajectory time cap was exceeded without a recorded extension.')
    gate = assess(rows, policy, complete=True)
    setup = load(out / 'core-setup-costs.json')
    fixture_manifest = load(out / 'scope-readiness-manifest.json')
    if len(fixture_manifest) != len(setup):
        raise ValueError('A scope preparation lost its auxiliary check.')
    fixture_calls = 0
    for index, prepared in enumerate(setup, 1):
        name = hashlib.sha256(prepared['scope'].encode()).hexdigest() + f"-{prepared['pid']}-{index}.json"
        path = out / 'scope-readiness' / name
        if fixture_manifest[name] != digest(path):
            raise ValueError('Scope fixture changed.')
        fixture = load(path)
        if (fixture['protocol_sha256'] != protocol_hash or fixture['scope'] != prepared['scope']
                or fixture['index_id'] != prepared['index_id'] or fixture['status'] != 'passed'
                or len(fixture['calls']) != len(policy['workload'])):
            raise ValueError('Scope fixture identity or workload changed.')
        fixture_calls += len(fixture['calls'])
    status = load(out / 'core-status.json')
    if status['status'] != 'completed' or status['completed'] != TOTAL or status['protocol_sha256'] != protocol_hash:
        raise ValueError('Runner did not complete this protocol.')
    invocations = sorted((out / 'invocations').iterdir())
    first_started = min(load(p)['started_at'] for p in invocations)
    seconds = (datetime.fromisoformat(status['updated_at']) - datetime.fromisoformat(first_started)).total_seconds()
    report = {'status': 'passed', 'checked_at': utc(), 'audit_source_sha256': digest(__file__),
              'protocol_sha256': protocol_hash, 'attempts': len(rows),
              'primary_attempts': sum(r['schedule']['repetition'] in PRIMARY_REPETITIONS for r in rows),
              'repeat_attempts': sum(r['schedule']['repetition'] not in PRIMARY_REPETITIONS for r in rows),
              'independent_questions': {ds: c for ds, c in COUNTS.items()},
              'scenarios': len(scenarios), 'fresh_browsecomp_scenarios': sum(sid not in pilot_scenarios for sid in scenarios),
              'provider_requests': request_count, 'trace_checks': trace_checks,
              'harness_errors': harness_errors, 'timing_contamination_flags': contaminated,
              'tools': {'calls': sum(g['calls'] for g in gate['groups']), 'valid_calls': sum(g['valid_calls'] for g in gate['groups']),
                        'input_errors': sum(g['input_errors'] for g in gate['groups']),
                        'operational_errors': sum(g['unexpected_errors'] for g in gate['groups'])},
              'tool_groups': gate['groups'], 'pause_reasons_on_replay': reasons, 'reviewed_attempts': reviewed,
              'guard': guard.state(), 'scope_preparations': len(setup), 'retained_scope_checks': len(fixture_manifest),
              'scope_fixture_calls': fixture_calls, 'measured_trajectory_seconds': sum(r['elapsed_ms'] for r in rows) / 1000,
              'setup_seconds': sum(s['elapsed_ms'] for s in setup) / 1000, 'invocations': len(invocations),
              'wall_seconds_first_invocation_to_completion': seconds, 'attempt_checksums': checksums,
              'artifact_checksums': {n: digest(out / n) for n in ('core-accounting.json', 'core-status.json',
                                                                   'core-setup-costs.json', 'scope-readiness-manifest.json')},
              'quality_scored': False, 'core_dispatch_authorized': protocol['core_dispatch_authorized']}
    write_json(destination, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output.resolve(), args.report.resolve())
    print(json.dumps({k: v for k, v in result.items() if k not in ('attempt_checksums', 'artifact_checksums', 'tool_groups')}, indent=2))


if __name__ == '__main__':
    main()

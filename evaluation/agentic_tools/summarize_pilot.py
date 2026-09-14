"""Offline accounting and cost estimates; no answer-score-based protocol tuning."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path

import numpy as np
from arkb.evaluation.external import digest, write_json
from .contract import ARM_BY_ID, OPTIONS, evidence_sets
from .selection import attempt_key


def audit_record(row, *, options=OPTIONS, think=True):
    report = (row.get('result') or {}).get('observation')
    if report is None:
        if row.get('error') is None:
            raise ValueError('Missing trace without a recorded execution failure.')
        return {'observed': False, 'checks': 0}
    arm = ARM_BY_ID[row['schedule']['arm']]
    models = report['models']
    if len(models) > (1 if arm.fixed else 8):
        raise ValueError('Model request budget exceeded.')
    for m in models:
        req = m['request']
        if req['options'] != options or req['think'] is not think or req['truncate'] is not False or req['shift'] is not False:
            raise ValueError('Actual model request differs from shared protocol.')
        if not think and ((m.get('response') or {}).get('message') or {}).get('thinking'):
            raise ValueError('Provider emitted thinking under the nonthinking protocol.')
    executed = [e for e in report['tools'] if e['executed'] and e['name'] != 'finish']
    if len(executed) > 12 or sum(e['name'] in ('search', 'match') for e in executed) > 10 or sum(e['name'] == 'read' for e in executed) > 6:
        raise ValueError('Executed tool budget exceeded.')
    if report['evidence']['delivered_tokens'] > 8000:
        raise ValueError('Evidence allowance exceeded.')
    for event in report['tools']:
        # Forbidden attempts are retained/charged; only successful dispatches
        # must match the capability set. A rejection is not arm escape.
        if event['status'] == 'success':
            if event['name'] == 'match' and not arm.match:
                raise ValueError('Forbidden match executed successfully.')
            if event['name'] == 'search' and (event['arguments'].get('mode') or arm.default_mode) not in arm.modes:
                raise ValueError('Forbidden search mode executed successfully.')
        expected = event['delivered_to_conversation'] and any(m['turn'] > event['turn'] for m in models)
        if event['submitted_to_model'] != expected:
            raise ValueError('Submitted-evidence flags disagree with provider requests.')
    if evidence_sets(report) != row['evidence_sources']:
        raise ValueError('Derived evidence coverage changed on replay.')
    return {'observed': True, 'checks': len(models) + len(report['tools']) + 4}


def failure_category(row):
    result = row.get('result') or {}
    if result.get('stop_reason') not in (None, 'error'):
        return None
    report = result.get('observation') or {}
    error = report.get('error') or row.get('error') or {}
    if error.get('type') == 'FileNotFoundError' and "'rg'" in error.get('message', ''):
        return 'runtime_dependency_missing'
    models = report.get('models') or []
    response = (models[-1].get('response') or {}) if models else {}
    if response.get('done_reason') == 'length':
        return 'output_token_limit'
    message = response.get('message') or {}
    if models and not message.get('content') and not message.get('tool_calls'):
        return 'empty_final_content'
    if 'reference' in str(error).lower():
        return 'invalid_final_reference'
    return 'other_execution_error'


def summarize(output):
    protocol_hash = digest(output / 'protocol.json')
    design = json.loads((output / 'design.json').read_text())
    scheduled = json.loads((output / 'pilot-schedule.json').read_text())
    core = json.loads((output / 'core-schedule.json').read_text())
    counts = Counter((r['dataset'], r['arm']) for r in core)
    groups = defaultdict(list)
    records, checks = [], 0
    artifact_bytes = {}
    for planned in scheduled:
        key = attempt_key(protocol_hash, planned)
        directory = output / 'pilot-attempts' / key
        if not (directory / 'complete.json').exists():
            continue
        complete = json.loads((directory / 'complete.json').read_text())
        if complete['result_sha256'] != digest(directory / 'result.json') or complete['provider_sha256'] != digest(directory / 'provider.jsonl'):
            raise ValueError('Immutable attempt hash mismatch.')
        row = json.loads((directory / 'result.json').read_text())
        if row['key'] != key or row['schedule'] != planned or row['protocol_sha256'] != protocol_hash:
            raise ValueError('Attempt key or scheduled identity mismatch.')
        checks += audit_record(row, options=design['options'], think=design['think'])['checks']
        records.append(row)
        artifact_bytes[key] = sum((directory / name).stat().st_size for name in
                                  ('attempt.json', 'result.json', 'provider.jsonl', 'complete.json'))
        groups[(planned['dataset'], planned['arm'])].append(row)
    metrics = []
    for (dataset, arm), rows in groups.items():
        times = [r['elapsed_ms'] / 1000 for r in rows]
        sizes = [artifact_bytes[r['key']] for r in rows]
        statuses = Counter((r.get('result') or {}).get('final', {}).get('status', 'error') for r in rows)
        stops = Counter((r.get('result') or {}).get('observation', {}).get('budget_stop_reason') or
                        (r.get('result') or {}).get('stop_reason', 'error') for r in rows)
        metrics.append({'dataset': dataset, 'arm': arm, 'pilot_attempts': len(rows),
                        'seconds_mean': float(np.mean(times)), 'seconds_p50': float(np.median(times)),
                        'seconds_p95': float(np.quantile(times, .95)),
                        'core_attempts': counts[(dataset, arm)],
                        'projected_core_seconds': float(np.mean(times)) * counts[(dataset, arm)],
                        'pilot_artifact_bytes': sum(sizes),
                        'projected_core_artifact_bytes': float(np.mean(sizes)) * counts[(dataset, arm)],
                        'final_statuses': dict(statuses), 'stop_reasons': dict(stops),
                        'model_requests': sum(len((r.get('result') or {}).get('observation', {}).get('models', [])) for r in rows)})
    complete = len(records) == 98
    summary = {'schema': 'agentic-tools-pilot-accounting-v1', 'status': 'complete' if complete else 'partial',
               'attempts_accounted': len(records), 'attempts_required': 98, 'trace_checks': checks,
               'protocol_sha256': protocol_hash, 'analysis_sha256': digest(__file__), 'groups': metrics,
               'projected_core_inference_hours': sum(x['projected_core_seconds'] for x in metrics) / 3600 if complete else None,
               'pilot_artifact_bytes': sum(artifact_bytes.values()),
               'projected_core_artifact_gib': sum(x['projected_core_artifact_bytes'] for x in metrics) / 2**30 if complete else None,
               'estimate_excludes': ['index setup', 'judge inference and retries', 'human review', 'archival'],
               'quality_scores_used_for_estimate': False, 'core_dispatch_ready': False,
               'failure_categories': dict(Counter(c for r in records if (c := failure_category(r)))),
               'failed_attempts': [{'key': r['key'], 'dataset': r['schedule']['dataset'], 'arm': r['schedule']['arm'],
                                    'category': c} for r in records if (c := failure_category(r))],
               'cost_estimate_scope': 'Observed protocol including its failures; an amended healthy protocol requires a new pilot estimate.',
               'remaining_core_gate': ['technical review of all pilot attempts', 'final core protocol',
                                       'judge calibration or explicit provisional review qualification']}
    write_json(output / 'pilot-accounting.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = summarize(a.output)
    print(json.dumps({k: report[k] for k in ('status', 'attempts_accounted', 'trace_checks', 'projected_core_inference_hours')}))


if __name__ == '__main__':
    main()

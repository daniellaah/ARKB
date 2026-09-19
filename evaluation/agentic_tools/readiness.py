"""Versioned operational checks. Answer accuracy is never a dispatch criterion."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from time import perf_counter

from .contract import ARM_BY_ID


POLICY = {
    'schema': 'arkb-tool-readiness-v1',
    'max_unexpected_errors': 0,
    'timely_completion_min': 0.99,
    'timely_seconds': 30,
    'p95_seconds_exclusive': 15,
    'percentile': 'nearest_rank',
    'online_stop': 'any unexpected tool failure or executed valid call taking at least 30 seconds; finish and retain the current trajectory, then pause',
    'input_error_codes': ['invalid_arguments', 'invalid_pattern', 'invalid_reference',
                          'undelivered_reference', 'invalid_citation', 'unknown_tool',
                          'capability_not_available'],
    'invalid_call_policy': 'Report separately, retain in trajectory denominator; source_unavailable is an input error only for read(source=...) without ref.',
    'latency_scope': 'prepared calls grouped by dataset and effective tool; report dataset/arm/tool as well; no inference of population reliability from a small pilot',
    'workload': [
        {'name': 'match', 'arguments': {'query': 'arkb_probe_absent_f5ebd93a78c1', 'limit': 20}, 'expect_empty': True},
        {'name': 'match', 'arguments': {'query': 'the', 'limit': 20}},
        {'name': 'match', 'arguments': {'query': 'arkb_probe_absent_f5ebd93a78c1', 'regex': True, 'limit': 20}, 'expect_empty': True},
        {'name': 'match', 'arguments': {'query': 'the', 'regex': True, 'limit': 20}},
        {'name': 'search', 'arguments': {'query': 'the', 'mode': 'bm25', 'limit': 20}},
        {'name': 'search', 'arguments': {'query': 'history of science and technology university research', 'mode': 'bm25', 'limit': 20}},
        {'name': 'search', 'arguments': {'query': 'arkb_probe_absent_f5ebd93a78c1', 'mode': 'bm25', 'limit': 20}},
        {'name': 'search', 'arguments': {'query': 'history of science and technology university research', 'mode': 'semantic', 'limit': 20}},
        {'name': 'search', 'arguments': {'query': 'history of science and technology university research', 'mode': 'hybrid', 'limit': 20}},
        {'name': 'read', 'arguments': {}, 'source_rule': 'first source in DocumentAccess deterministic order'},
    ],
}


def tool_kind(event, arm):
    if event['name'] == 'search':
        return 'search:' + ((event.get('arguments') or {}).get('mode') or ARM_BY_ID[arm].default_mode)
    return event['name']


def is_input_error(event, policy):
    code = (event.get('error') or {}).get('code')
    if event.get('status') != 'recoverable_error':
        return False
    if code in policy['input_error_codes']:
        return True
    args = event.get('arguments') or {}
    return code == 'source_unavailable' and event['name'] == 'read' and bool(args.get('source')) and not args.get('ref')


def call_rows(record, policy):
    observation = (record.get('result') or {}).get('observation') or {}
    schedule = record['schedule']
    rows = []
    for event in observation.get('tools', []):
        if not event.get('executed'):
            continue
        seconds = event.get('elapsed_ms')
        valid_time = type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0
        rows.append({'attempt': record['key'], 'dataset': schedule['dataset'], 'arm': schedule['arm'],
                     'tool': tool_kind(event, schedule['arm']), 'index': event['index'],
                     'seconds': seconds / 1000 if valid_time else None,
                     'input_error': is_input_error(event, policy), 'status': event['status'],
                     'error_code': (event.get('error') or {}).get('code') or (event.get('error') or {}).get('type')})
    return rows


def operational_failures(record, policy):
    return [r for r in call_rows(record, policy) if not r['input_error'] and
            (r['status'] != 'success' or r['seconds'] is None or r['seconds'] >= policy['timely_seconds'])]


def execution_failure(record):
    report = (record.get('result') or {}).get('observation')
    if record.get('error') or not report:
        return True
    if record.get('competing_processes_after'):
        return True
    # Invalid model output is a measured model outcome. A failed provider
    # request or harness deadline is an operational failure requiring review.
    return any(m['status'] == 'fatal_error' and (m.get('error') or {}).get('stage') == 'model_request'
               for m in report.get('models', []))


def metrics(rows, policy):
    valid = [r for r in rows if not r['input_error']]
    times = sorted(r['seconds'] for r in valid if r['seconds'] is not None)
    unexpected = sum(r['status'] != 'success' or r['seconds'] is None for r in valid)
    timely = sum(r['status'] == 'success' and r['seconds'] is not None and r['seconds'] < policy['timely_seconds'] for r in valid)
    p95 = times[math.ceil(len(times) * .95) - 1] if times else None
    rate = timely / len(valid) if valid else None
    passed = bool(valid) and unexpected <= policy['max_unexpected_errors'] and rate >= policy['timely_completion_min'] and p95 < policy['p95_seconds_exclusive']
    return {'calls': len(rows), 'valid_calls': len(valid), 'input_errors': len(rows) - len(valid),
            'unexpected_errors': unexpected, 'timely_completion_rate': rate, 'seconds_p95': p95,
            'seconds_mean': sum(times) / len(times) if times else None,
            'error_codes': dict(Counter(r['error_code'] for r in rows if r['error_code'])),
            'passed': passed}


def assess(records, policy, *, complete):
    rows = [r for record in records for r in call_rows(record, policy)]
    groups, arms = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['tool']].append(row)
        arms[row['dataset'], row['arm'], row['tool']].append(row)
    pooled = [{'dataset': d, 'tool': t, **metrics(v, policy)} for (d, t), v in sorted(groups.items())]
    # Groups containing only model input errors are reported, but cannot establish
    # backend readiness; mandatory scope fixtures independently cover every tool.
    failures = [r['key'] for r in records if execution_failure(r)]
    passed = complete and bool(rows) and not failures and all(g['passed'] for g in pooled if g['valid_calls'])
    return {'status': 'passed' if passed else ('failed' if complete else 'pending'),
            'quality_scores_used': False, 'groups': pooled,
            'by_arm': [{'dataset': d, 'arm': a, 'tool': t, **metrics(v, policy)} for (d, a, t), v in sorted(arms.items())],
            'calls': len(rows), 'execution_failures': failures, 'core_dispatch_authorized': False}


def run_scope_checks(tools, dataset, scope, policy):
    """Deterministic synthetic queries over the complete prepared scope."""
    rows = []
    for index, fixture in enumerate(policy['workload']):
        args = dict(fixture['arguments'])
        if fixture.get('source_rule'):
            args['source'] = next(iter(tools._documents._paths())).name
        start = perf_counter()
        result, error = None, None
        try:
            from arkb.evaluation.deadline import evaluation_deadline
            with evaluation_deadline(policy['timely_seconds']):
                result = getattr(tools, fixture['name'])(**args)
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
        seconds = perf_counter() - start
        # Verify evidence against live files after timing. No answer labels.
        verified = 0
        if error is None:
            try:
                hits = [result['result']] if fixture['name'] == 'read' else result['results']
                if fixture.get('expect_empty') and hits:
                    raise ValueError('Registered absent probe unexpectedly matched.')
                for hit in hits:
                    doc = tools._documents.read(hit['document_id'], source=hit['source'],
                                                start_char=hit['start_char'], end_char=hit['end_char'])
                    if doc.content != hit['content'] or doc.document_revision != hit['document_revision']:
                        raise ValueError('Returned evidence differs from the frozen live source.')
                    verified += 1
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
        rows.append({'dataset': dataset, 'scope': scope, 'index': index,
                     'tool': tool_kind({'name': fixture['name'], 'arguments': args}, 'A-All'),
                     'arguments': args, 'seconds': seconds, 'input_error': False,
                     'status': 'error' if error else 'success', 'error_code': error['type'] if error else None,
                     'error': error, 'live_reference_checks': verified,
                     'result_sha256': hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest() if result else None})
    groups = defaultdict(list)
    for row in rows:
        groups[row['tool']].append(row)
    summaries = [{'tool': t, **metrics(v, policy)} for t, v in sorted(groups.items())]
    return {'scope': scope, 'dataset': dataset, 'status': 'passed' if all(g['passed'] for g in summaries) else 'failed',
            'calls': rows, 'groups': summaries, 'quality_scores_used': False}

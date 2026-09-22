"""Protocol reliability accounting, separate from retrieval/answer quality."""
from collections import Counter
from statistics import mean
import json


def summarize_reliability(rows):
    if not rows:
        raise ValueError('Reliability summary requires attempted runs.')
    reports = [(r.get('result') or {}).get('observation', {}) for r in rows]
    # Finish is a control operation with a model-turn allowance. Report it
    # separately so adding the finish tool does not inflate retrieval cost.
    requested = [e for report in reports for e in report.get('tools', []) if e['name'] != 'finish']
    attempts = [e for e in requested if e['executed'] or e['status'] in ('error','fatal_error')]
    def validation(event):
        detail = event.get('error') or {}
        return (detail.get('code') in {'invalid_arguments','invalid_reference','invalid_pattern',
                    'stale_reference','source_unavailable','unknown_tool','undelivered_reference'}
                or detail.get('type') in {'ValueError','TypeError','LookupError','DocumentNotFound','ValidationError'})
    def failed(event):
        return event['status'] in ('error','fatal_error','recoverable_error')
    exact = [e for e in attempts if e['name'] == 'match']
    finals = [r for r in rows if (r.get('result') or {}).get('response') is not None and r['stop_reason']=='final']
    mu = [r for r in rows if r['track']=='musique']
    mu_valid = [r for r in mu if r['parse_error'] is None and r['stop_reason']=='final']
    canonical = [(r.get('result') or {}).get('final') for r in rows]
    envelopes = 0
    for final in canonical:
        if final is None: continue
        json.dumps(final,allow_nan=False)
        if (final.get('schema_version')=='arkb-agent-final-v1'
                and final.get('status') in ('answered','partial','insufficient_evidence','error')
                and isinstance(final.get('citations'),list)
                and isinstance(final.get('termination_reason'),str)):
            envelopes += 1
    rate = lambda count,total: {'count':count,'denominator':total,'rate':count/total if total else None}
    return {
        'runs':len(rows), 'collection_tool_validation_errors':rate(sum(validation(e) for e in attempts),len(attempts)),
        'fatal_tool_errors':rate(sum(e['status'] in ('error','fatal_error') for e in attempts),len(attempts)),
        'recoverable_tool_errors':rate(sum(e['status']=='recoverable_error' for e in attempts),len(attempts)),
        'runs_with_final_output':rate(len(finals),len(rows)),
        'musique_structured_prediction_validity':rate(len(mu_valid),len(mu)),
        'canonical_envelope_validity':rate(envelopes,len(rows)) if any(canonical) else None,
        'exact_match_errors':rate(sum(failed(e) for e in exact),len(exact)),
        'exact_match_timeouts':rate(sum((e.get('error') or {}).get('code')=='exact_timeout' or
            (e.get('error') or {}).get('type') in ('EvaluationDeadlineExceeded','ExactTimeout','TimeoutExpired') for e in exact),len(exact)),
        'average_requested_collection_calls':len(requested)/len(rows),
        'average_attempted_collection_calls':len(attempts)/len(rows),
        'average_finish_calls':sum(e['name']=='finish' for p in reports for e in p.get('tools',[]))/len(rows),
        'finish_validation_errors':sum(e['status']=='recoverable_error' for p in reports for e in p.get('tools',[]) if e['name']=='finish'),
        'average_model_turns':mean(len(p.get('models',[])) for p in reports),
        'average_elapsed_ms':mean(r['elapsed_ms'] for r in rows),
        'stop_reasons':dict(Counter(r['stop_reason'] for r in rows)),
        'termination_reasons':dict(Counter(((r.get('result') or {}).get('final') or {}).get('termination_reason',r['stop_reason']) for r in rows)),
        'limits':['Rates use attempted collection calls, excluding finish; requested-but-skipped calls remain in requested cost.',
                  'Old validation classification uses recorded exception types; new classification uses explicit error codes.',
                  'Bright had no old structured-output requirement; MuSiQue is the common structured-validity denominator.',
                  'Canonical error envelopes count as serializable, but never as successful final answers.'],
    }

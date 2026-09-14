"""Scoring-side pure metrics; no imports from inference runners."""
from collections import Counter

from arkb.agent.state import AgentFinal
from arkb.evaluation.multihop import canonical_prediction, answer_f1, support_f1, _normalize
from .contract import ARM_BY_ID


def canonical(row):
    value = (row.get('result') or {}).get('final')
    return AgentFinal(**value) if value else None


def answer_eligible(row):
    final = canonical(row)
    return bool(final and final.status in ('answered', 'partial') and
                isinstance(final.answer, str) and final.answer.strip())


def musique_row(row, gold):
    final = canonical(row)
    prediction, error = canonical_prediction(final, gold['source_map'])
    prediction['id'] = gold['id']
    valid = error is None
    aliases = [gold['answer'], *gold['answer_aliases']]
    af = answer_f1(prediction['predicted_answer'] or '', aliases) if valid else 0.0
    em = float(any(_normalize(prediction['predicted_answer'] or '') == _normalize(a) for a in aliases)) if valid else 0.0
    sf = support_f1(prediction['predicted_support_idxs'], gold['support_idxs']) if valid else 0.0
    return {'prediction': prediction, 'gold_answerable': gold['answerable'],
            'answer_f1': af if gold['answerable'] else None,
            'answer_em': em if gold['answerable'] else None,
            'support_f1': sf if gold['answerable'] else None,
            'answerability_correct': float(prediction['predicted_answerable'] is gold['answerable']),
            'execution_valid': valid}


def evidence_coverage(row, labels):
    qid = row['schedule']['id']
    source_map = labels['source_map']
    groups = row.get('evidence_sources') or {k: [] for k in ('returned', 'delivered', 'submitted')}
    scores = {}
    for label_name in ('qrels', 'gold_qrels'):
        if label_name not in labels:
            continue
        judged = labels[label_name][qid]
        positive = {str(k) for k, value in judged.items() if value > 0}
        for stage, sources in groups.items():
            if any(source not in source_map for source in sources):
                raise ValueError('Observed source is outside the frozen corpus mapping.')
            observed = {str(source_map[source]) for source in sources}
            scores[f'{label_name}_{stage}'] = {
                'positive_recall': len(observed & positive) / len(positive) if positive else None,
                'positive_hits': len(observed & positive), 'positive_total': len(positive),
                'observed_sources': len(observed),
                'unjudged_sources': len(observed - set(judged)),
            }
    return scores


def costs_and_behavior(row):
    report = (row.get('result') or {}).get('observation') or {}
    tools = report.get('tools') or []
    counts = Counter()
    completed_legs = Counter()
    uncertain_searches = 0
    search_modes, source_sets = [], []
    for tool in tools:
        if not tool['executed']:
            continue
        name = tool['name']
        counts[name] += 1
        if name == 'search':
            mode = tool['arguments'].get('mode') or ARM_BY_ID[row['schedule']['arm']].default_mode
            search_modes.append(mode)
            if tool['status'] == 'success':
                for leg in (('bm25', 'semantic') if mode == 'hybrid' else (mode,)):
                    completed_legs[leg] += 1
            elif tool['status'] == 'fatal_error':
                uncertain_searches += 1
        raw = tool.get('raw_result') or {}
        hits = [raw['result']] if 'result' in raw else raw.get('results', [])
        source_sets.append({h['source'] for h in hits})
    seen, repeated, new = set(), 0, 0
    for sources in source_sets:
        repeated += len(sources & seen)
        new += len(sources - seen)
        seen.update(sources)
    return {'elapsed_ms': row['elapsed_ms'], 'model_requests': len(report.get('models') or []),
            'tools_executed': dict(counts), 'completed_retrieval_legs': dict(completed_legs),
            'search_calls_with_unknown_partial_leg_work': uncertain_searches,
            'retrieval_leg_measurement': 'Completed legs derived from successful calls in the frozen sequential engine; partial failed searches are unknown, never counted as zero work.',
            'search_modes': search_modes,
            'mode_transitions': sum(a != b for a, b in zip(search_modes, search_modes[1:])),
            'new_sources_across_calls': new, 'repeated_sources_across_calls': repeated,
            'usage': report.get('usage'), 'evidence_tokens': report.get('evidence'),
            'model_load_nanoseconds': sum((m.get('usage') or {}).get('load_duration') or 0 for m in report.get('models', [])),
            'stop_reason': (row.get('result') or {}).get('stop_reason', 'error'),
            'budget_stop_reason': report.get('budget_stop_reason')}

"""Offline core scoring, registered statistics and blinded citation-review export."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from arkb.evaluation.external import digest, write_json
from .contract import ARMS
from .records import read_records
from .scoring import canonical, musique_row, evidence_coverage, costs_and_behavior
from .statistics import paired_comparisons, query_units, session_variability
from .summarize_pilot import audit_record


def citation_sample(rows, selected_ids, scenarios):
    selected = set(selected_ids[:20])
    cases = [r for r in rows if r['schedule']['dataset'] == 'browsecomp-plus'
             and r['schedule']['id'] in selected and r['schedule']['repetition'] == 0]
    if (len(selected) != 20 or len(cases) != 140
            or {(r['schedule']['id'], r['schedule']['arm']) for r in cases} != {(q, a.id) for q in selected for a in ARMS}):
        raise ValueError('The citation audit requires 20 complete queries across all seven arms.')
    review, key = [], {}
    for row in cases:
        blind = hashlib.sha256(('citation-review-v1|' + row['key']).encode()).hexdigest()
        final = canonical(row)
        review.append({'review_id': blind, 'question': scenarios[row['scenario_id']]['query'],
                       'answer': final.answer if final else None, 'status': final.status if final else 'error',
                       'cited_excerpts': final.citations if final else [],
                       'claims': None, 'reviewer': None, 'review_status': 'pending'})
        key[blind] = {'key': row['key'], 'schedule': row['schedule']}
    return sorted(review, key=lambda x: x['review_id']), key


def musique_pairs(rows):
    groups = defaultdict(list)
    for row in rows:
        s = row['schedule']
        groups[(s['id'], s['arm'], s['repetition'])].append(row)
    pairs = []
    for values in groups.values():
        if len(values) != 2 or {v['musique']['gold_answerable'] for v in values} != {True, False}:
            raise ValueError('An original MuSiQue pair is incomplete.')
        answerable = next(v for v in values if v['musique']['gold_answerable'])
        correct_pair = all(v['musique']['answerability_correct'] for v in values)
        s = {**answerable['schedule'], 'variant': 'v0'}
        pairs.append({'schedule': s,
                      'answer_f1': answerable['musique']['answer_f1'],
                      'answer_em': answerable['musique']['answer_em'],
                      'support_f1': answerable['musique']['support_f1'],
                      'answerability': sum(v['musique']['answerability_correct'] for v in values) / 2,
                      'group_answer_sufficiency_f1': answerable['musique']['answer_f1'] if correct_pair else 0.0,
                      'group_support_sufficiency_f1': answerable['musique']['support_f1'] if correct_pair else 0.0})
    return pairs


def first_positive_discovery(row, labels):
    positives = {str(k) for k, v in labels['qrels'][row['schedule']['id']].items() if v > 0}
    answer = {stage: None for stage in ('returned', 'delivered', 'submitted')}
    queries = 0
    for event in ((row.get('result') or {}).get('observation') or {}).get('tools', []):
        if event['executed'] and event['name'] in ('search', 'match'):
            queries += 1
        raw = event.get('raw_result') or {}
        hits = [raw['result']] if 'result' in raw else raw.get('results', [])
        delivered = set(event.get('delivered_refs', [h['ref'] for h in hits]))
        for hit in hits:
            if str(labels['source_map'][hit['source']]) not in positives:
                continue
            for stage, present in [('returned', True),
                                   ('delivered', event['delivered_to_conversation'] and hit['ref'] in delivered),
                                   ('submitted', event['submitted_to_model'] and hit['ref'] in delivered)]:
                if present and answer[stage] is None:
                    answer[stage] = {'tool_index': event['index'], 'model_turn': event['turn'], 'query_calls_so_far': queries}
    return answer


def registered_analysis(protocol):
    """Repetition roles come from the frozen protocol, never from observed outcomes."""
    statistics = protocol.get('statistics') or {}
    primary = tuple(statistics.get('primary_repetitions', (0, 1, 2)))
    subset = statistics.get('repeat_subset')
    if subset:
        subset = {'dataset': subset['dataset'], 'ids': set(subset['ids']), 'repetitions': tuple(subset['repetitions'])}
    return primary, subset


def analyze(output, grading, destination):
    protocol, original = read_records(output)
    if protocol['phase'] != 'core' or len(original) != protocol['core_attempts']:
        raise ValueError('A complete registered core run is required.')
    primary_repetitions, repeat_subset = registered_analysis(protocol)
    design = json.loads((output / 'design.json').read_text())
    checks = sum(audit_record(r, options=design['options'], think=design['think'])['checks'] for r in original)
    grade_protocol = json.loads((grading / 'protocol.json').read_text())
    if grade_protocol['source_protocol_sha256'] != digest(output / 'protocol.json'):
        raise ValueError('Grading belongs to a different inference protocol.')
    keys = json.loads((grading / 'private-review-key.json').read_text())
    grades = {keys[g['id']]['key']: g for g in json.loads((grading / 'scores.json').read_text())}
    if len(grades) != sum(r['schedule']['dataset'] == 'browsecomp-plus' for r in original):
        raise ValueError('Every core BrowseComp attempt needs a scoring status.')
    labels = {ds: json.loads((output / 'scoring' / (ds + '.json')).read_text())
              for ds in ('browsecomp-plus', 'fiqa', 'nfcorpus', 'musique')}
    scored = []
    for row in original:
        ds = row['schedule']['dataset']
        record = {'key': row['key'], 'scenario_id': row['scenario_id'], 'schedule': row['schedule'],
                  'final_status': canonical(row).status if canonical(row) else 'error',
                  'costs': costs_and_behavior(row), 'latency_isolated_at_checked_boundaries': not bool(row.get('competing_processes_after'))}
        if ds == 'musique':
            gold = labels[ds][row['scenario_id']]
            record['musique'] = musique_row(row, gold)
            evidence_labels = {'source_map': {k: str(v) for k, v in gold['source_map'].items()},
                               'qrels': {row['schedule']['id']: {str(v): 1 for v in gold['support_idxs']}}}
        else:
            evidence_labels = labels[ds]
        record['evidence'] = evidence_coverage(row, evidence_labels)
        record['first_positive_discovery'] = first_positive_discovery(row, evidence_labels)
        if ds == 'browsecomp-plus':
            record['answer_success'] = grades[row['key']]['score']
        scored.append(record)
    destination.mkdir(exist_ok=True)
    write_json(destination / 'scores.json', scored)
    summaries = {}
    for ds in ('browsecomp-plus', 'fiqa', 'nfcorpus', 'musique'):
        rows = [r for r in scored if r['schedule']['dataset'] == ds and r['schedule']['repetition'] in primary_repetitions]
        arms = {}
        for arm in ARMS:
            subset = [r for r in rows if r['schedule']['arm'] == arm.id]
            times = [r['costs']['elapsed_ms'] / 1000 for r in subset]
            clean = [r['costs']['elapsed_ms'] / 1000 for r in subset if r['latency_isolated_at_checked_boundaries']]
            arms[arm.id] = {'attempts': len(subset), 'final_statuses': dict(Counter(r['final_status'] for r in subset)),
                            'elapsed_seconds_mean': float(np.mean(times)), 'elapsed_seconds_p50': float(np.median(times)),
                            'elapsed_seconds_p95': float(np.quantile(times, .95)),
                            'model_requests': sum(r['costs']['model_requests'] for r in subset),
                            'timing_contamination_flags': len(times) - len(clean),
                            'isolated_elapsed_seconds_mean': float(np.mean(clean)) if clean else None,
                            'isolated_elapsed_seconds_p95': float(np.quantile(clean, .95)) if clean else None,
                            'tools_executed': dict(sum((Counter(r['costs']['tools_executed']) for r in subset), Counter())),
                            'completed_retrieval_legs': dict(sum((Counter(r['costs']['completed_retrieval_legs']) for r in subset), Counter())),
                            'searches_with_unknown_partial_leg_work': sum(r['costs']['search_calls_with_unknown_partial_leg_work'] for r in subset),
                            'first_positive_found': {stage: sum(r['first_positive_discovery'][stage] is not None for r in subset)
                                                     for stage in ('returned', 'delivered', 'submitted')},
                            'canonical_final_subset': {'attempts': sum(r['final_status'] != 'error' for r in subset),
                                'evidence_positive_recall': {metric: float(np.mean(defined)) if (defined := [r['evidence'][metric]['positive_recall']
                                    for r in subset if r['final_status'] != 'error' and r['evidence'][metric]['positive_recall'] is not None]) else None
                                    for metric in subset[0]['evidence']}},
                            'usage_incomplete_requests': {k: sum(((r['costs']['usage'] or {}).get(k) or {}).get('requests', r['costs']['model_requests'])
                                    - ((r['costs']['usage'] or {}).get(k) or {}).get('defined_requests', 0) for r in subset)
                                    for k in ('prompt_eval_count', 'eval_count')},
                            'token_usage_known': {k: sum(((r['costs']['usage'] or {}).get(k) or {}).get('known_total', 0)
                                                          for r in subset) for k in ('prompt_eval_count', 'eval_count')}}
        ratios = {}
        for arm in ARMS:
            if arm.id == 'A-All':
                continue
            left, right = arms['A-All'], arms[arm.id]
            lv, rv = left['isolated_elapsed_seconds_mean'], right['isolated_elapsed_seconds_mean']
            ratios[arm.id] = {'isolated_latency_ratio': lv / rv if lv is not None and rv else None,
                **{k + '_ratio': left['token_usage_known'][k] / right['token_usage_known'][k]
                   if not left['usage_incomplete_requests'][k] and not right['usage_incomplete_requests'][k] and right['token_usage_known'][k]
                   else None for k in ('prompt_eval_count', 'eval_count')}}
        summary = {'arms': arms, 'attempts': len(rows), 'primary_repetitions': list(primary_repetitions),
                   'evidence_comparisons': {}, 'a_all_cost_ratios': ratios}
        for metric in sorted(rows[0]['evidence']):
            metric_rows = [{**r, 'metric': r['evidence'][metric]['positive_recall']} for r in rows]
            units = query_units(metric_rows, 'metric', variants=('v0', 'v1') if ds == 'musique' else ('v0',),
                                repetitions=primary_repetitions)
            summary['evidence_comparisons'][metric] = paired_comparisons(units)
        if ds == 'browsecomp-plus':
            summary['answer_success'] = paired_comparisons(query_units(rows, 'answer_success', repetitions=primary_repetitions), primary=True)
            summary['quality_status'] = 'provisional_pending_independent_human_review'
        if ds == 'musique':
            pairs = musique_pairs(rows)
            summary['musique_metrics'] = {metric: paired_comparisons(query_units(pairs, metric, repetitions=primary_repetitions)) for metric in
                ('answer_f1', 'answer_em', 'support_f1', 'answerability', 'group_answer_sufficiency_f1', 'group_support_sufficiency_f1')}
            summary['evidence_scope'] = 'Available labeled support in each supplied variant context; not all missing multi-hop facts.'
        if repeat_subset and repeat_subset['dataset'] == ds:
            subset = [r for r in scored if r['schedule']['dataset'] == ds and r['schedule']['id'] in repeat_subset['ids']]
            if len(subset) != len(repeat_subset['ids']) * len(ARMS) * len(repeat_subset['repetitions']):
                raise ValueError('The registered repeat subset is incomplete.')
            summary['repeat_subset'] = {'attempts': len(subset), 'questions': len(repeat_subset['ids']),
                                        'repetitions': list(repeat_subset['repetitions']),
                                        'answer_success': session_variability(subset, 'answer_success', repetitions=repeat_subset['repetitions'])
                                        if ds == 'browsecomp-plus' else None,
                                        'final_status_identical_all_repetitions': sum(
                                            len({r['final_status'] for r in subset if r['schedule']['id'] == q and r['schedule']['arm'] == a.id}) == 1
                                            for q in repeat_subset['ids'] for a in ARMS),
                                        'cells': len(repeat_subset['ids']) * len(ARMS),
                                        'scope': 'Registered before inference; excluded from primary contrasts; no best-of-k.'}
        summaries[ds] = summary
    selection = json.loads((output / 'selection.json').read_text())
    scenarios = json.loads((output / 'inference/scenarios.json').read_text())
    review, review_key = citation_sample(original, selection['tracks']['browsecomp-plus'][0]['core'], scenarios)
    write_json(destination / 'citation-review-pending.json', review)
    write_json(destination / 'private-citation-review-key.json', review_key)
    summary = {'status': 'automated_analysis_complete_review_pending', 'quality_status': 'provisional',
               'datasets': summaries, 'registered_attempts': len(scored),
               'primary_attempts': sum(r['schedule']['repetition'] in primary_repetitions for r in scored),
               'analysis_source_sha256': digest(__file__), 'trace_checks': checks,
               'pending_judge_scores': sum(r.get('answer_success') is None for r in scored if r['schedule']['dataset'] == 'browsecomp-plus'),
               'citation_review': {'responses': len(review), 'status': 'pending', 'supported_claim_fraction': None,
                                   'unsupported_response_rate': None, 'empty_abstained_separate': True},
               'no_cross_dataset_pooled_score': True, 'source_protocol_sha256': digest(output / 'protocol.json')}
    write_json(destination / 'summary.json', summary)
    write_json(destination / 'analysis-checksums.json', {name: digest(destination / name) for name in
               ('scores.json', 'summary.json', 'citation-review-pending.json', 'private-citation-review-key.json')})
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--grading', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.output, args.grading, args.destination)
    print(json.dumps({k: result[k] for k in ('status', 'registered_attempts', 'trace_checks', 'pending_judge_scores')}))


if __name__ == '__main__':
    main()

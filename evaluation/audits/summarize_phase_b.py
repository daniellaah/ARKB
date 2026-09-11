"""Scoring-side Phase B audit; gold never enters the reranker or its input builder."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from arkb.evaluation.external import (aspect_metrics, digest, rank_metrics, read_jsonl,
                                      reference_metrics, verify_checksums, write_json)


DATASETS = ('scifact', 'bright-stackoverflow', 'bright-robotics')


def metrics(labels, qid, ranking):
    return {**rank_metrics(labels['qrels'][qid], ranking),
            **(aspect_metrics(labels['aspects'][qid], ranking)
               if qid in labels['aspects'] else {})}


def paired(left, right):
    result = {}
    for key in left[0]:
        values = np.asarray([b[key] - a[key] for a, b in zip(left, right)])
        rng = np.random.default_rng(20260911)
        means = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
        result[key] = {'mean_delta': float(values.mean()),
                       'ci95': np.quantile(means, [.025, .975]).tolist(),
                       'wins': int((values > 0).sum()), 'ties': int((values == 0).sum()),
                       'losses': int((values < 0).sum())}
    return result


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {'count': len(values), 'mean': float(values.mean()),
            'min': float(values.min()), 'max': float(values.max()),
            **dict(zip(('p10', 'p50', 'p90'), np.quantile(values, [.1, .5, .9]).tolist()))}


def input_summary(rows):
    candidates = [d for r in rows for d in r.get('inputs', [])]
    if not candidates:
        return None
    count = len(candidates)
    return {'candidates': count,
            'truncation_rate': sum(d['truncated'] for d in candidates) / count,
            'zero_body_rate': sum(d['body_tokens_retained'] == 0 for d in candidates) / count,
            'body_below_tokens_rate': {str(k): sum(d['body_tokens_retained'] < k for d in candidates) / count
                                       for k in (32, 64, 128)},
            'query_tokens_before': distribution([r['inputs'][0]['query_tokens_before'] for r in rows]),
            'query_tokens_retained': distribution([r['inputs'][0]['query_tokens_retained'] for r in rows]),
            'title_tokens_before': distribution([d['title_tokens_before'] for d in candidates]),
            'title_tokens_retained': distribution([d['title_tokens_retained'] for d in candidates]),
            'body_tokens_before': distribution([d['body_tokens_before'] for d in candidates]),
            'body_tokens_retained': distribution([d['body_tokens_retained'] for d in candidates]),
            'all_body_empty_queries': sum(all(d['body_tokens_retained'] == 0 for d in r['inputs']) for r in rows),
            'half_body_empty_queries': sum(sum(d['body_tokens_retained'] == 0 for d in r['inputs']) * 2 >= len(r['inputs']) for r in rows),
            'queries_over_512_tokens': [r['qid'] for r in rows if r['inputs'][0]['query_tokens_before'] > 512]}


def audit_variant(pools, labels, rows, *, tie_policy='stable'):
    if [r['qid'] for r in rows] != [p['qid'] for p in pools]:
        raise ValueError('Variant must preserve the full registered query order.')
    scored = []; references = {}; hybrid_scores = []; old_scores = []
    for pool, row in zip(pools, rows):
        qid = pool['qid']; before = pool['hybrid_ranking']; ranking = row['ranking']
        if (len(ranking) != len(set(ranking)) or set(ranking) != set(before)
                or ranking[20:] != before[20:] or set(ranking[:20]) != set(before[:20])):
            raise ValueError('Candidate set or tail changed: ' + qid)
        m = metrics(labels, qid, ranking); h = metrics(labels, qid, before)
        b = metrics(labels, qid, pool['baseline_ranking'])
        if h != labels['hybrid_metrics'][qid] or b != labels['baseline_metrics'][qid]:
            raise ValueError('Frozen metrics do not reproduce P4.')
        if m['recall@100'] != h['recall@100']:
            raise ValueError('Recall@100 changed.')
        scores = row.get('scores', [])
        if scores and (len(scores) != 20 or not all(type(s) in (int, float) and np.isfinite(s) for s in scores)):
            raise ValueError('Invalid saved scores must be explicitly excluded from usable scores.')
        if row.get('fallback'):
            if ranking != before:
                raise ValueError('Fallback did not preserve the complete Hybrid order.')
        elif scores:
            def key(i):
                if tie_policy == 'identity':
                    hit = pool['candidates'][i]
                    return -scores[i], (hit['source_id'], 'chunk', hit['chunk_id'])
                return -scores[i], i
            order = sorted(range(20), key=key)
            expected = [pool['candidate_document_ids'][i] for i in order] + before[20:]
            if ranking != expected:
                raise ValueError('Scores do not justify the recorded ranking under the declared tie policy.')
        else:
            raise ValueError('Missing scores without an explicit fallback.')
        unique = len(set(scores)) if scores else 0
        gold = {d for d, gain in labels['qrels'][qid].items() if gain > 0}
        positions = {d: i+1 for i, d in enumerate(before)}
        after = {d: i+1 for i, d in enumerate(ranking)}
        promotions = [d for d in gold & set(before[:20]) if after[d] < positions[d]]
        demotions = [d for d in gold & set(before[:20]) if after[d] > positions[d]]
        lost10 = sorted(gold & set(before[:10]) - set(ranking[:10]))
        diagnostics = row.get('inputs', [])
        insufficient = [pool['candidate_document_ids'][i] for i, d in enumerate(diagnostics)
                        if d['body_tokens_retained'] < 32]
        categories = []
        if gold - set(before): categories.append('positive_absent_from_hybrid100')
        if gold & (set(before) - set(before[:20])): categories.append('positive_outside_pool20')
        if demotions: categories.append('positive_demoted_by_reranker')
        if lost10: categories.append('positive_demoted_out_of_top10')
        if set(lost10) & set(insufficient): categories.append('demotion_with_body_below_32_tokens_observational')
        if row.get('tie_only_changed'): categories.append('ranking_changed_only_by_tie_policy')
        if row.get('fallback'): categories.append('ranking_preserved_by_fallback')
        scored.append({'qid': qid, 'metrics': m, 'hybrid_metrics': h, 'baseline_metrics': b,
                       'hybrid_top10': before[:10], 'hybrid_top20': before[:20], 'hybrid_top100': before,
                       'reranker_input_pool': pool['candidate_document_ids'], 'reranker_output_top10': ranking[:10],
                       'known_positive_ids': sorted(gold), 'aspects': labels['aspects'].get(qid, []),
                       'score_ties': bool(scores) and unique < len(scores),
                       'all_equal_scores': bool(scores) and unique == 1, 'unique_scores': unique,
                       'mean_absolute_rank_displacement': sum(abs(after[d] - positions[d]) for d in before[:20])/20,
                       'positive_promotions': sorted(promotions), 'positive_demotions': sorted(demotions),
                       'categories': categories, 'fallback': row.get('fallback'),
                       'elapsed_ms': row.get('elapsed_ms'),
                       'strata': {'query_length': ('le128' if diagnostics[0]['query_tokens_before'] <= 128 else
                                 '129to192' if diagnostics[0]['query_tokens_before'] <= 192 else
                                 '193to512' if diagnostics[0]['query_tokens_before'] <= 512 else 'gt512') if diagnostics else None,
                                  'any_truncation': any(d['truncated'] for d in diagnostics) if diagnostics else None,
                                  'any_zero_body': any(d['body_tokens_retained'] == 0 for d in diagnostics) if diagnostics else None}})
        references[qid] = ranking; hybrid_scores.append(h); old_scores.append(b)
    reference = reference_metrics(labels['qrels'], references)
    maximum_error = max(abs(r['metrics'][k]-v) for r in scored for k, v in reference[r['qid']].items())
    if maximum_error > 1e-9:
        raise ValueError('Independent trec_eval disagreement.')
    mean = {k: sum(r['metrics'][k] for r in scored)/len(scored) for k in scored[0]['metrics']}
    summary = {'queries': len(rows), 'metrics': mean,
               'versus_hybrid': paired(hybrid_scores, [r['metrics'] for r in scored]),
               'versus_B0': paired(old_scores, [r['metrics'] for r in scored]),
               'reference_max_error': maximum_error,
               'queries_with_score_ties': sum(r['score_ties'] for r in scored),
               'all_equal_score_queries': sum(r['all_equal_scores'] for r in scored),
               'mean_unique_score_count': sum(r['unique_scores'] for r in scored)/len(scored),
               'mean_absolute_rank_displacement': sum(r['mean_absolute_rank_displacement'] for r in scored)/len(scored),
               'relevant_document_promotions': sum(len(r['positive_promotions']) for r in scored),
               'relevant_document_demotions': sum(len(r['positive_demotions']) for r in scored),
               'fallback_activation_rate': sum(bool(r['fallback']) for r in scored)/len(scored),
               'fallback_counts': dict(Counter(r['fallback'] for r in scored if r['fallback'])),
               'latency_ms_per_query': distribution([r['elapsed_ms'] for r in scored if r['elapsed_ms'] is not None]),
               'latency_ms_per_candidate': distribution([r['elapsed_ms']/20 for r in scored if r['elapsed_ms'] is not None]),
               'inputs': input_summary(rows),
               'nonexclusive_category_counts': dict(Counter(c for r in scored for c in r['categories']))}
    strata = {}
    for name in ('query_length', 'any_truncation', 'any_zero_body', 'score_ties'):
        groups = {}
        for r in scored:
            value = r[name] if name == 'score_ties' else r['strata'][name]
            if value is not None: groups.setdefault(str(value), []).append(r)
        strata[name] = {key: {'queries': len(group), 'versus_hybrid': paired(
                [r['hybrid_metrics'] for r in group], [r['metrics'] for r in group])}
                       for key, group in groups.items()}
    summary['diagnostic_strata'] = strata
    return summary, scored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--variant', required=True, help='B0-saved, B1, or an inference directory name')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify_checksums(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output/'audit.py')
    result = {'schema': 'arkb-phase-b-analysis-v1', 'variant': args.variant,
              'audit_sha256': digest(Path(__file__)),
              'pool_manifest_sha256': digest(args.run/'manifest.json'),
              'bootstrap': {'seed': 20260911, 'resamples': 10000, 'unit': 'query', 'multiple_comparison_adjustment': False},
              'category_counts_overlap': True, 'release_eligible': False, 'datasets': {}}
    for ds in DATASETS:
        pools = read_jsonl(args.run/'pools'/f'{ds}.jsonl')
        labels = json.loads((args.run/'scoring'/f'{ds}.json').read_text())
        if args.variant in ('B0-saved', 'B1'):
            rows = []
            for p in pools:
                ranking = p['baseline_ranking']
                if args.variant == 'B1':
                    order = sorted(range(20), key=lambda i: -p['baseline_scores'][i])
                    ranking = [p['candidate_document_ids'][i] for i in order] + p['hybrid_ranking'][20:]
                rows.append({'qid': p['qid'], 'scores': p['baseline_scores'], 'ranking': ranking,
                             'tie_only_changed': ranking != p['baseline_ranking'],
                             'elapsed_ms': p['baseline_stage_ms']})
            with (args.output/f'{ds}-replay.jsonl').open('x') as f:
                for r in rows: f.write(json.dumps(r)+'\n')
        else:
            verify_checksums(args.run/args.variant)
            rows = read_jsonl(args.run/args.variant/f'{ds}.jsonl')
        if args.variant in ('B3', 'B4'):
            prepared = read_jsonl(args.run/'allocations'/args.variant/f'{ds}.jsonl')
            if len(prepared) != len(rows): raise ValueError('Registered inputs are incomplete.')
            for row, expected in zip(rows, prepared):
                if row['qid'] != expected['qid'] or len(row['inputs']) != len(expected['inputs']):
                    raise ValueError('Registered input identity mismatch.')
                for actual, original in zip(row['inputs'], expected['inputs']):
                    if {k: v for k, v in actual.items() if k != 'score'} != original:
                        raise ValueError('Actual model input or diagnostic differs from preregistration.')
        summary, scored = audit_variant(pools, labels, rows,
            tie_policy='identity' if args.variant in ('B0-saved', 'B0-inference') else 'stable')
        summary['changed_from_B0_rankings'] = sum(r['ranking'] != p['baseline_ranking'] for r, p in zip(rows, pools))
        result['datasets'][ds] = summary
        with (args.output/f'{ds}-attribution.jsonl').open('x') as f:
            for row in scored: f.write(json.dumps(row)+'\n')
        print(ds, summary['metrics'], flush=True)
    write_json(args.output/'summary.json', result)
    write_json(args.output/'checksums.json', {p.name: digest(p) for p in args.output.iterdir() if p.is_file()})


if __name__ == '__main__':
    main()

"""Apply the registered shared-policy rule to complete, audited Phase B results."""

import argparse
import json
from pathlib import Path
import shutil

from arkb.evaluation.external import digest, read_jsonl, verify_checksums, write_json
from summarize_phase_b import DATASETS, paired


VARIANTS = ('B0', 'B1', 'B2', 'B3', 'B4')
PRIMARY = ('ndcg@10', 'aspect_recall@10')


def select(summaries):
    """No per-domain policy choice; the fallback tie-break was registered in advance."""
    eligible = [v for v in ('B3', 'B4') if summaries[v]['scifact']['metrics']['ndcg@10']
                - summaries['B2']['scifact']['metrics']['ndcg@10'] >= -.02]
    def dominates(left, right):
        differences = [summaries[left][ds]['metrics'][m] - summaries[right][ds]['metrics'][m]
                       for ds in DATASETS[1:] for m in PRIMARY]
        return all(d >= 0 for d in differences) and any(d > 0 for d in differences)
    if not eligible:
        selected, reason = 'B2', 'Neither allocation passes the SciFact development regression guard.'
    elif len(eligible) == 1:
        selected, reason = eligible[0], 'Only this allocation passes the SciFact development regression guard.'
    elif dominates('B3', 'B4'):
        selected, reason = 'B3', 'B3 Pareto-dominates B4 on both primary metrics in both technical domains.'
    elif dominates('B4', 'B3'):
        selected, reason = 'B4', 'B4 Pareto-dominates B3 on both primary metrics in both technical domains.'
    else:
        selected, reason = 'B4', 'No Pareto dominance; the registered tie-break retains more query context.'
    gates = {ds: {'ndcg_lower_ci_positive': summaries[selected][ds]['versus_hybrid']['ndcg@10']['ci95'][0] > 0,
                  'aspect_mean_nonnegative': summaries[selected][ds]['versus_hybrid']['aspect_recall@10']['mean_delta'] >= 0}
             for ds in DATASETS[1:]}
    quality_gate = selected in eligible and all(all(g.values()) for g in gates.values())
    return {'eligible_allocations': eligible, 'selected_variant': selected, 'selection_reason': reason,
            'technical_domain_gates': gates, 'default_quality_gate_passed': quality_gate,
            'recommendation': ('latency judgment required before a default recommendation' if quality_gate
                               else 'reranking should remain optional'),
            'depth_study_gate_passed': quality_gate,
            'release_eligible': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify_checksums(args.run); verify_checksums(args.run/'allocations')
    summaries = {}; rows = {}; sources = {}
    for variant in VARIANTS:
        folder = args.run/'final-analyses'/variant
        verify_checksums(folder)
        summary = json.loads((folder/'summary.json').read_text())
        if summary['pool_manifest_sha256'] != digest(args.run/'manifest.json'):
            raise ValueError('Analysis uses a different frozen candidate pool.')
        summaries[variant] = summary['datasets']; rows[variant] = {}
        sources[variant] = digest(folder/'summary.json')
        for ds in DATASETS:
            records = read_jsonl(folder/f'{ds}-attribution.jsonl')
            expected = read_jsonl(args.run/'pools'/f'{ds}.jsonl')
            if [r['qid'] for r in records] != [p['qid'] for p in expected]:
                raise ValueError('Analysis query order differs from the frozen pool.')
            rows[variant][ds] = records
    decision = select(summaries)
    decision.update(schema='arkb-phase-b-selection-v1',
                    protocol_sha256=digest(args.run/'allocations/protocol.json'),
                    analysis_summary_hashes=sources,
                    bootstrap={'resamples': 10000, 'seed': 20260911, 'unit': 'query',
                               'multiple_comparison_adjustment': False},
                    paired_comparisons={})
    for left, right in (('B0', 'B1'), ('B1', 'B2'), ('B2', 'B3'), ('B2', 'B4'), ('B3', 'B4')):
        decision['paired_comparisons'][f'{right}_versus_{left}'] = {
            ds: paired([r['metrics'] for r in rows[left][ds]], [r['metrics'] for r in rows[right][ds]])
            for ds in DATASETS}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output/'select.py')
    shutil.copyfile(Path(__file__).with_name('summarize_phase_b.py'), args.output/'summarize_phase_b.py')
    write_json(args.output/'decision.json', decision)
    write_json(args.output/'checksums.json', {p.name: digest(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in decision.items() if k != 'paired_comparisons'}, indent=2))


if __name__ == '__main__':
    main()

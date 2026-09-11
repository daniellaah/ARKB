"""Verify the selected production policy and replay its frozen scores through B5.

The selected B3/B4 arm already performed full real inference. This final pass
checks every production input and ranking without another quality trial. One
fixed first query per dataset is re-scored to check batching equivalence.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

from arkb.evaluation.external import digest, read_jsonl, verify_checksums, write_json
from arkb.retrieval.models import SearchResult
from arkb.retrieval.qwen_rerank import QwenRerankerScorer
from arkb.retrieval.rerank import Reranker
from run_phase_b import DATASETS, ROOT, emit, environment, model_files, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run = args.run.resolve()
    verify_checksums(run); verify_checksums(args.selection)
    decision = json.loads((args.selection/'decision.json').read_text())
    selected = decision['selected_variant']
    if selected not in ('B3', 'B4'):
        raise ValueError('This finalization requires a validated explicit allocation.')
    verify_checksums(run/selected); verify_checksums(run/'allocations')
    original = json.loads((run/selected/'experiment.json').read_text())
    if original['status'] != 'completed':
        raise ValueError('Selected full inference must be complete.')
    protocol = json.loads((run/'allocations/protocol.json').read_text())
    if model_files()[1] != protocol['model_files']:
        raise ValueError('Pinned model files changed.')
    import torch
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output/'runner.py')
    shutil.copytree(ROOT/'src', args.output/'measured-source/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    meta = {'schema': 'arkb-phase-b-final-v1', 'status': 'running', 'started_at': now(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'selected_variant': selected, 'selection_sha256': digest(args.selection/'decision.json'),
            'pool_manifest_sha256': digest(run/'manifest.json'), 'environment': environment(),
            'new_quality_trials': 0, 'sanity_queries': 'First registered query per dataset, 20 candidates each.',
            'latency_source': 'Selected complete B3/B4 inference; replay overhead is recorded separately.',
            'hypothesis': 'Final production tokens and score ordering reproduce the selected trial; '
                          'exact-all-equal fallback preserves Hybrid order and score provenance.',
            'tie_policy': 'stable incoming order', 'fallback_policy': 'all_equal_scores or empty_body; '
                          'invalid scores and expected execution failures preserve original ranking/scores',
            'stopping_rule': 'One full input/score replay and three fixed batching checks; no retuning.',
            'datasets': {}}
    write_json(args.output/'protocol.json', meta)
    scorer = QwenRerankerScorer(cache_folder=str(ROOT/'.arkb/models'), local_files_only=True)
    policy = protocol['variants'][selected]
    if (scorer.query_cap, scorer.title_cap, scorer.query_strategy, scorer.max_length, scorer.batch_size) != (
            policy['query_cap'], policy['title_cap'], policy['query_strategy'], 512, 16):
        raise ValueError('Production defaults do not match the selected policy.')
    meta['scorer_identity'] = scorer.identity
    try:
        for ds in DATASETS:
            pools = read_jsonl(run/'pools'/f'{ds}.jsonl')
            rows = read_jsonl(run/selected/f'{ds}.jsonl')
            if [p['qid'] for p in pools] != [r['qid'] for r in rows]:
                raise ValueError('Incomplete selected trial.')
            stats = {'queries': len(pools), 'inputs_verified': 0, 'all_equal_score_queries': 0,
                     'all_body_empty_queries': 0, 'rankings_equal_to_selected': 0}
            with (args.output/f'{ds}.jsonl').open('x') as stream:
                for i, (pool, row) in enumerate(zip(pools, rows)):
                    hits = tuple(SearchResult(**h) for h in pool['candidates'])
                    details = scorer.prepare_inputs(pool['query'], hits)
                    expected = [{k: v for k, v in d.items() if k not in ('score', 'original_hybrid_rank')}
                                for d in row['inputs']]
                    if details != expected:
                        raise ValueError('Final production model input differs from the measured trial.')
                    stats['inputs_verified'] += len(details)
                    empty = all(d['body_empty'] for d in details)
                    stats['all_body_empty_queries'] += empty
                    stats['all_equal_score_queries'] += len(set(row['scores'])) == 1
                    if i == 0:
                        scores = scorer.score(pool['query'], hits)
                        difference = max(abs(a-b) for a, b in zip(scores, row['scores']))
                        if len(scores) != 20 or difference > 1e-4:
                            raise ValueError('Production batching check differs from the measured model scores.')
                        stats['fixed_batching_check'] = {'qid': pool['qid'], 'scores': scores,
                                                        'max_absolute_score_difference': difference}
                    class SavedScorer:
                        identity = scorer.identity
                        score_type = scorer.score_type
                        def score(self, query, candidates):
                            if query != pool['query'] or tuple(candidates) != hits:
                                raise ValueError('Score replay candidate mismatch.')
                            if empty:
                                from arkb.retrieval.rerank import EmptyRerankerInput
                                raise EmptyRerankerInput('No candidate retains document body tokens.')
                            return row['scores']
                    start = perf_counter()
                    result = Reranker(SavedScorer()).rerank(pool['query'], hits)
                    replay_ms = (perf_counter()-start)*1000
                    source_to_doc = {h.source: d for h, d in zip(hits, pool['candidate_document_ids'])}
                    ranking = [source_to_doc[h.source] for h in result] + pool['hybrid_ranking'][20:]
                    fallback = result[0].metadata['rerank'].get('fallback')
                    if ranking != row['ranking']:
                        raise ValueError('B5 unexpectedly changes a selected stable ranking.')
                    stats['rankings_equal_to_selected'] += 1
                    emit(stream, {**row, 'ranking': ranking, 'fallback': fallback,
                                  'result_provenance': [h.metadata['rerank'] for h in result],
                                  'replay_ms': replay_ms, 'final_results': [asdict(h) for h in result]})
            meta['datasets'][ds] = stats
            print(ds, stats, flush=True)
        if model_files()[1] != protocol['model_files']:
            raise ValueError('Pinned model files changed during final verification.')
        meta['status'] = 'completed'
    except BaseException as error:
        meta.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        meta['finished_at'] = now()
        write_json(args.output/'experiment.json', meta)
        write_json(args.output/'checksums.json', {p.relative_to(args.output).as_posix(): digest(p)
            for p in args.output.rglob('*') if p.is_file() and p.name != 'checksums.json'})


if __name__ == '__main__':
    main()

"""Freeze P4 candidates and execute registered downstream-only reranker trials."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

from arkb.evaluation.external import digest, load_external, read_jsonl, verify_checksums, write_json
from arkb.knowledge.models import Chunk, ChunkRecord
from arkb.retrieval.models import SearchResult


DATASETS = ('scifact', 'bright-stackoverflow', 'bright-robotics')
ROOT = Path(__file__).resolve().parents[2]


def emit(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    stream.flush()


def now():
    return datetime.now(timezone.utc).isoformat()


def environment():
    import torch
    return {'python': sys.version, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
            'packages': {n: version(n) for n in ('torch', 'transformers', 'tokenizers', 'numpy',
                                                'pytrec-eval-terrier')},
            'torch_threads': torch.get_num_threads(),
            'threads': {n: os.environ.get(n) for n in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'TOKENIZERS_PARALLELISM')}}


def model_files():
    from arkb.retrieval.qwen_rerank import QWEN_MODEL, QWEN_REVISION
    directory = ROOT / '.arkb/models' / ('models--' + QWEN_MODEL.replace('/', '--')) / 'snapshots' / QWEN_REVISION
    return directory, {p.relative_to(directory).as_posix(): digest(p)
                       for p in sorted(directory.rglob('*')) if p.is_file()}


def freeze(out):
    """Labels are copied only to scoring/. pools/ contains model inputs and provenance."""
    out.mkdir(parents=True, exist_ok=False)
    for name in ('pools', 'scoring', 'p4-provenance', 'baseline-source'):
        (out / name).mkdir()
    shutil.copytree(ROOT / 'src', out / 'baseline-source/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('pyproject.toml', 'uv.lock'):
        shutil.copyfile(ROOT / name, out / 'baseline-source' / name)
    directory, files = model_files()
    if not any(n.endswith('.safetensors') for n in files):
        raise ValueError('Pinned model weights are unavailable.')
    manifest = {'schema': 'arkb-phase-b-pools-v1', 'created_at': now(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'phase_a_production_commit': 'c47a015', 'model_files': files,
                'datasets': {}, 'release_eligible': False}
    for ds in DATASETS:
        run = ROOT / 'evaluation/results' / f'p4-{ds}-indexed'
        data_path = ROOT / 'evaluation/results/p4-data' / ds
        checked = verify_checksums(run)
        data = load_external(data_path)
        protocol = json.loads((run / 'protocol.json').read_text())
        experiment = json.loads((run / 'experiment.json').read_text())
        if experiment['status'] != 'completed' or digest(data_path / 'manifest.json') != protocol['dataset_manifest_sha256']:
            raise ValueError('Unaccepted P4 input.')
        old_files = experiment.get('reranker_files_before')
        if old_files is not None and old_files != files:
            raise ValueError('Pinned reranker artifacts drifted.')
        for name in ('qwen_rerank.py', 'rerank.py'):
            if digest(run / 'measured-source/src/arkb/retrieval' / name) != digest(out / 'baseline-source/src/arkb/retrieval' / name):
                raise ValueError('Phase A reranker differs from measured P4: ' + name)
        dest = out / 'p4-provenance' / ds
        dest.mkdir()
        for name in ('protocol.json', 'experiment.json', 'summary.json', 'checksums.json', 'rows.jsonl'):
            shutil.copyfile(run / name, dest / name)
        shutil.copyfile(data_path / 'manifest.json', dest / 'data-manifest.json')
        mapping = data.source_map()
        notes = {n.source: n for n in data.notes()}
        arms = {q: {} for q in data.queries}
        for row in read_jsonl(run / 'rows.jsonl'):
            arms[row['qid']][row['arm']] = row
        source_rows = read_jsonl(run / 'candidates.jsonl')
        if [r['qid'] for r in source_rows] != list(data.queries):
            raise ValueError('Candidate query order differs.')
        counts = {'candidates': 0, 'nonzero_chunk_index': 0, 'validated_source_spans': 0}
        with (out / 'pools' / f'{ds}.jsonl').open('x') as stream:
            for raw in source_rows:
                qid = raw['qid']; pool = raw['rerank_inputs']
                hybrid = arms[qid]['hybrid']; old = arms[qid]['hybrid-rerank20']
                if hybrid['error'] or old['error'] or len(pool) != 20:
                    raise ValueError('Incomplete candidate pool.')
                ids = [mapping[h['source']] for h in pool]
                if ids != hybrid['ranking'][:20] or len(set(ids)) != 20:
                    raise ValueError('Source aggregation differs.')
                scores = {h['source']: h['score'] for h in raw['rerank_scores']}
                if set(scores) != {h['source'] for h in pool}:
                    raise ValueError('Missing saved scores.')
                for h in pool:
                    hit = SearchResult(**h); m = hit.metadata
                    chunk = Chunk(content=hit.content, title=m['title'], source=hit.source,
                                  start_char=hit.start_char, end_char=hit.end_char,
                                  **{k: m[k] for k in ('chunk_index', 'heading_path', 'section_id',
                                     'section_start_char', 'section_end_char', 'occurrence')})
                    record = ChunkRecord.from_note(chunk, note=notes[hit.source], vault_id=ds)
                    if (record.document_id, record.document_revision, record.chunk_id) != (
                            hit.source_id, m['document_revision'], hit.chunk_id):
                        raise ValueError('Candidate identity mismatch.')
                    counts['nonzero_chunk_index'] += m['chunk_index'] != 0
                    counts['validated_source_spans'] += 1
                baseline = sorted(range(20), key=lambda i: (-scores[pool[i]['source']], SearchResult(**pool[i]).identity))
                ranking = [ids[i] for i in baseline] + hybrid['ranking'][20:]
                if ranking != old['ranking']:
                    raise ValueError('Saved B0 scores fail exact ranking replay.')
                emit(stream, {'dataset': ds, 'qid': qid, 'query': data.queries[qid],
                              'hybrid_ranking': hybrid['ranking'], 'candidates': pool,
                              'candidate_document_ids': ids, 'hybrid_legs': raw['legs'],
                              'baseline_scores': [scores[h['source']] for h in pool],
                              'baseline_ranking': ranking, 'baseline_stage_ms': raw['stage_ms']['hybrid-rerank20']})
                counts['candidates'] += len(pool)
        write_json(out / 'scoring' / f'{ds}.json', {'qrels': data.qrels, 'aspects': data.aspects,
                   'hybrid_metrics': {q: arms[q]['hybrid']['metrics'] for q in data.queries},
                   'baseline_metrics': {q: arms[q]['hybrid-rerank20']['metrics'] for q in data.queries}})
        manifest['datasets'][ds] = {'query_ids': list(data.queries), 'counts': counts,
                'dataset_manifest_sha256': digest(data_path / 'manifest.json'),
                'p4_candidate_sha256': digest(run / 'candidates.jsonl'),
                'pool_sha256': digest(out / 'pools' / f'{ds}.jsonl'),
                'p4_files_checked': checked,
                'historical_model_file_hashes_available': old_files is not None}
        print(ds, counts, flush=True)
    # A local lossless tokenizer copy makes input reconstruction independent of model downloads.
    shutil.copytree(directory, out / 'tokenizer',
                    ignore=shutil.ignore_patterns('*.safetensors', 'README.md'))
    write_json(out / 'manifest.json', manifest)
    write_json(out / 'checksums.json', {p.relative_to(out).as_posix(): digest(p)
               for p in out.rglob('*') if p.is_file()})


def baseline(out):
    verify_checksums(out)
    manifest = json.loads((out / 'manifest.json').read_text())
    if model_files()[1] != manifest['model_files']:
        raise ValueError('Model files differ from frozen inputs.')
    trial = out / 'B0-inference'
    trial.mkdir(exist_ok=False)
    shutil.copyfile(__file__, trial / 'runner.py')
    spec = importlib.util.spec_from_file_location('phase_b_baseline_qwen',
                out / 'baseline-source/src/arkb/retrieval/qwen_rerank.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(4)
    meta = {'status': 'running', 'started_at': now(), 'environment': environment(),
            'git_commit': manifest['git_commit'], 'runner_sha256': digest(Path(__file__)),
            'pool_manifest_sha256': digest(out / 'manifest.json'),
            'input_policy': 'P4 whole-sequence right truncation', 'sequence_limit': 512,
            'rerank_depth': 20, 'tie_policy': 'document/chunk identity', 'fallback_policy': 'none',
            'primary_metrics': ['ndcg@10', 'aspect_recall@10'],
            'secondary_metrics': ['recall@10', 'recall@100', 'alpha_ndcg@10'],
            'latency': 'per-query scorer wall time including tokenization; extra saved-input inspection excluded',
            'hypothesis': 'Unchanged model/input/batching reproduces P4 scores and rankings.',
            'score_absolute_tolerance': 1e-4, 'ranking_tolerance': 0,
            'stopping_rule': 'One pass over all frozen queries; retain and characterize deviations before B3/B4.'}
    write_json(trial / 'protocol.json', meta)
    started = perf_counter()
    scorer = module.QwenRerankerScorer(cache_folder=str(ROOT / '.arkb/models'), local_files_only=True)
    meta['load_ms'] = (perf_counter() - started) * 1000
    meta['scorer_identity'] = scorer.identity
    try:
        for ds in DATASETS:
            with (trial / f'{ds}.jsonl').open('x') as stream:
                rows = read_jsonl(out / 'pools' / f'{ds}.jsonl')
                for index, row in enumerate(rows):
                    hits = tuple(SearchResult(**h) for h in row['candidates'])
                    started = perf_counter(); scores = scorer.score(row['query'], hits)
                    latency = (perf_counter() - started) * 1000
                    inputs = scorer._inputs(row['query'], hits)
                    tokens = [ids[mask.bool()].tolist() for ids, mask in
                              zip(inputs['input_ids'], inputs['attention_mask'])]
                    order = sorted(range(len(hits)), key=lambda i: (-scores[i], hits[i].identity))
                    ranking = [row['candidate_document_ids'][i] for i in order] + row['hybrid_ranking'][20:]
                    delta = max(abs(a-b) for a, b in zip(scores, row['baseline_scores']))
                    emit(stream, {'dataset': ds, 'qid': row['qid'], 'scores': scores, 'input_ids': tokens,
                                  'ranking': ranking, 'elapsed_ms': latency,
                                  'max_score_difference': delta, 'rank_reproduced': ranking == row['baseline_ranking']})
                    if index % 10 == 0 or index == len(rows)-1:
                        print('B0', ds, index+1, '/', len(rows), 'max_abs_score_error', delta, flush=True)
        if model_files()[1] != manifest['model_files'] or environment() != meta['environment']:
            raise ValueError('Model or software environment drift.')
        meta['status'] = 'completed'
    except BaseException as error:
        meta.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        meta['finished_at'] = now()
        write_json(trial / 'experiment.json', meta)
        write_json(trial / 'checksums.json', {p.name: digest(p) for p in trial.iterdir()
                                            if p.is_file() and p.name != 'checksums.json'})


def instrument(out):
    """B2 reuses B0 scores and verifies token identity against the legacy tokenizer call."""
    from transformers import AutoTokenizer
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    from arkb.retrieval.qwen_rerank import INSTRUCTION, PREFIX, SUFFIX
    verify_checksums(out)
    trial = out / 'B2-corrected'; trial.mkdir(exist_ok=False)
    shutil.copyfile(__file__, trial / 'runner.py')
    shutil.copytree(ROOT / 'src', trial / 'measured-source/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    tokenizer = AutoTokenizer.from_pretrained(str(out / 'tokenizer'), padding_side='left', local_files_only=True)
    builder = QwenInputBuilder(tokenizer, max_length=512)
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    write_json(trial / 'protocol.json', {'variant': 'B2', 'registered_at': now(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'pool_manifest_sha256': digest(out / 'manifest.json'),
        'input_policy': 'unchanged whole-sequence right truncation', 'sequence_limit': 512,
        'rerank_depth': 20, 'tie_policy': 'stable input order', 'fallback': 'none',
        'hypothesis': 'Instrumentation preserves every model input token and all B1 scores/rankings.',
        'new_model_calls': 0, 'latency_source': 'saved P4 scorer timings',
        'instrumentation_amendment': 'B2 initially assigned a separator-crossing token to an empty title. '
            'Require a nonempty title span; old B2 retained. Scores, model tokens and rankings never changed.',
        'stopping_rule': 'All 516 queries once; fail on any legacy token difference.'})
    for ds in DATASETS:
        rows = read_jsonl(out / 'pools' / f'{ds}.jsonl')
        with (trial / f'{ds}.jsonl').open('x') as stream:
            for row in rows:
                hits = [SearchResult(**h) for h in row['candidates']]
                details = [builder.prepare(row['query'], h) for h in hits]
                pairs = [f'<Instruct>: {INSTRUCTION}\n<Query>: {row["query"]}\n<Document>: '
                         f'{h.metadata.get("title") or ""}\n\n{h.content}' for h in hits]
                ids = tokenizer(pairs, padding=False, truncation='longest_first',
                    return_attention_mask=False, max_length=512-len(prefix)-len(suffix))['input_ids']
                if any(d['input_ids'] != prefix + raw + suffix for d, raw in zip(details, ids)):
                    raise ValueError('Instrumentation changed legacy tokens.')
                for i, d in enumerate(details):
                    d.update(original_hybrid_rank=i+1, score=row['baseline_scores'][i])
                order = sorted(range(len(hits)), key=lambda i: -row['baseline_scores'][i])
                emit(stream, {'qid': row['qid'], 'dataset': ds, 'inputs': details,
                    'scores': row['baseline_scores'], 'elapsed_ms': row['baseline_stage_ms'],
                    'ranking': [row['candidate_document_ids'][i] for i in order] + row['hybrid_ranking'][20:]})
        print('B2 token identity verified', ds, len(rows), flush=True)
    write_json(trial / 'checksums.json', {p.relative_to(trial).as_posix(): digest(p)
                                        for p in trial.rglob('*') if p.is_file()})


def register(out):
    """Predeclare two allocations and freeze their actual inputs before scoring."""
    from transformers import AutoTokenizer
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    verify_checksums(out)
    allocation = out / 'allocations'; allocation.mkdir(exist_ok=False)
    shutil.copytree(ROOT / 'src', allocation / 'measured-source/src',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copyfile(__file__, allocation / 'runner.py')
    tokenizer = AutoTokenizer.from_pretrained(str(out / 'tokenizer'), padding_side='left', local_files_only=True)
    manifest = json.loads((out / 'manifest.json').read_text())
    protocol = {'schema': 'arkb-phase-b-allocations-v1', 'registered_at': now(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'pool_manifest_sha256': digest(out / 'manifest.json'),
        'dataset_hashes': {k: v['dataset_manifest_sha256'] for k, v in manifest['datasets'].items()},
        'pool_hashes': {k: v['pool_sha256'] for k, v in manifest['datasets'].items()},
        'query_ids': {k: v['query_ids'] for k, v in manifest['datasets'].items()},
        'model_files': manifest['model_files'], 'sequence_limit': 512, 'rerank_depth': 20,
        'batch_size': 16, 'precision': 'float32', 'device': 'cpu', 'threads': 4,
        'tie_policy': 'stable incoming Hybrid order',
        'execution_fallback': 'invalid scores or RuntimeError/OSError during scoring preserve the input order and original scores; programming/configuration errors propagate',
        'primary_metrics': ['ndcg@10', 'aspect_recall@10'],
        'secondary_metrics': ['recall@10', 'recall@100', 'alpha_ndcg@10'],
        'paired_bootstrap': {'resamples': 10000, 'seed': 20260911, 'unit': 'query'},
        'variants': {
            'B3': {'query_cap': 128, 'title_cap': 64, 'query_strategy': 'head_tail',
                   'hypothesis': 'Preserving body evidence removes the catastrophic long-query failure.'},
            'B4': {'query_cap': 192, 'title_cap': 64, 'query_strategy': 'head_tail',
                   'hypothesis': 'Additional query context may improve ranking relative to B3 at a smaller body budget.'}},
        'allocation': 'Caps count actual joined-context field tokens, including the query gap marker. '
            'Preserve half of the capped query from each end, separated by a newline/ellipsis/newline. '
            'Clip complete Unicode character spans, rebuild the official labels and separators, and recount. '
            'Title retains its head. Body receives the remainder, with at least min(original body tokens,128). '
            'Unchanged short fields keep their exact text; no summaries or gold-dependent selection.',
        'hypothesis_basis': 'Query-only inspection: technical code/logs often occupy the middle; '
            '31/76 long Stack Overflow and 25/55 long Robotics queries contain a question mark in the final quarter. '
            'Neither this punctuation proxy nor the inspected examples imply that all tails are useful.',
        'inspection_sha256': digest(out / 'query-structure-inspection.json'),
        'latency': 'Per-query production Reranker wall time including input construction, excluding model load. '
            'B3/B4 order alternates per query; same CPU and four threads. No upstream retrieval.',
        'stopping_rule': 'One trial for every registered query in each variant. No parameter edits after score inspection. '
            'A technical failure is retained; only unexecuted rows may resume with a documented amendment.',
        'selection_rule': 'One policy across all datasets. Exclude allocations losing more than 0.02 mean SciFact nDCG@10 '
            'relative to B2 (a preregistered development regression guard, not a release margin). '
            'Among eligible policies, prefer a policy that weakly dominates the other on both nDCG@10 and '
            'weighted aspect recall in both BRIGHT domains, with at least one strict improvement. '
            'If neither dominates, prefer B4 to retain more query context; report the tradeoff. '
            'If neither is eligible, retain stable legacy reranking as an explicitly limited option.',
        'default_decision_rule': 'Propose a global default only if the selected shared policy has positive nDCG@10 '
            'paired lower 95% bounds on both BRIGHT domains, nonnegative mean weighted-aspect changes, '
            'passes the SciFact guard, and the measured latency is justified by the gains. '
            'Otherwise keep reranking optional and preserve current opt-in behavior. No dataset-specific routing.',
        'B5_rule': 'On the selected B3/B4 scores, replay exact-all-equal fallback separately. Stable ties already '
            'make this an identity ordering; no inference is required. All-body-empty fallback is a separate '
            'input-only replay; omit a new quality arm if its trigger never occurs. No variance threshold.',
        'B6_rule': 'Only if the selected policy establishes clear positive effects in both technical domains; '
            'otherwise omit the secondary depth study to avoid confounding remaining model/input limitations.',
        'release_eligible': False}
    write_json(allocation / 'protocol.json', protocol)
    for variant, policy in protocol['variants'].items():
        folder = allocation / variant; folder.mkdir()
        builder = QwenInputBuilder(tokenizer, max_length=512, **{k: policy[k] for k in
                                   ('query_cap', 'title_cap', 'query_strategy')})
        for ds in DATASETS:
            rows = read_jsonl(out / 'pools' / f'{ds}.jsonl')
            with (folder / f'{ds}.jsonl').open('x') as stream:
                for row in rows:
                    start = perf_counter()
                    details = [builder.prepare(row['query'], SearchResult(**h)) for h in row['candidates']]
                    elapsed = (perf_counter() - start)*1000
                    for i, d in enumerate(details):
                        d['original_hybrid_rank'] = i+1
                    emit(stream, {'qid': row['qid'], 'inputs': details, 'input_preparation_ms': elapsed})
            print('registered', variant, ds, len(rows), flush=True)
    write_json(allocation / 'checksums.json', {p.relative_to(allocation).as_posix(): digest(p)
                                              for p in allocation.rglob('*') if p.is_file()})


def execute_allocations(out):
    from arkb.retrieval.qwen_rerank import QwenRerankerScorer
    from arkb.retrieval.rerank import Reranker
    import torch
    torch.set_num_threads(4)
    verify_checksums(out); allocation = out / 'allocations'; verify_checksums(allocation)
    baseline_run = out / 'B0-inference'; verify_checksums(baseline_run)
    baseline_meta = json.loads((baseline_run / 'experiment.json').read_text())
    if baseline_meta['status'] != 'completed':
        raise ValueError('B0 inference must finish before allocation scoring.')
    for ds in DATASETS:
        rows = read_jsonl(baseline_run / f'{ds}.jsonl')
        pools = read_jsonl(out / 'pools' / f'{ds}.jsonl')
        if len(rows) != len(pools) or any(not r['rank_reproduced'] or r['max_score_difference'] > 1e-4 for r in rows):
            raise ValueError('B0 changed: characterize the variation before proceeding.')
    protocol = json.loads((allocation / 'protocol.json').read_text())
    if model_files()[1] != protocol['model_files']:
        raise ValueError('Model files changed.')
    def check_source():
        for path in (allocation/'measured-source/src').rglob('*.py'):
            if digest(path) != digest(ROOT/'src'/path.relative_to(allocation/'measured-source/src')):
                raise ValueError('Production code changed after preregistration.')
    check_source()
    folders = {v: out / v for v in protocol['variants']}
    for v, folder in folders.items():
        folder.mkdir(exist_ok=False)
        write_json(folder/'protocol.json', {**protocol, 'active_variant': v,
                   'allocation_manifest_sha256': digest(allocation/'checksums.json')})
    scorers = {}; meta = {}
    for v, policy in protocol['variants'].items():
        meta[v] = {'status': 'running', 'started_at': now(), 'environment': environment()}
        start = perf_counter()
        scorers[v] = QwenRerankerScorer(cache_folder=str(ROOT/'.arkb/models'), local_files_only=True,
            **{k: policy[k] for k in ('query_cap', 'title_cap', 'query_strategy')})
        meta[v].update(load_ms=(perf_counter()-start)*1000, scorer_identity=scorers[v].identity)
    try:
        for ds in DATASETS:
            pools = read_jsonl(out/'pools'/f'{ds}.jsonl')
            prepared = {v: read_jsonl(allocation/v/f'{ds}.jsonl') for v in folders}
            streams = {v: (folder/f'{ds}.jsonl').open('x') for v, folder in folders.items()}
            try:
                for i, pool in enumerate(pools):
                    hits = tuple(SearchResult(**h) for h in pool['candidates'])
                    for v in (('B3', 'B4') if i % 2 == 0 else ('B4', 'B3')):
                        inputs = scorers[v].prepare_inputs(pool['query'], hits)
                        expected = prepared[v][i]
                        if expected['qid'] != pool['qid'] or any(
                            d['input_ids'] != e['input_ids'] for d, e in zip(inputs, expected['inputs'])):
                            raise ValueError('Registered model inputs changed.')
                        start = perf_counter(); result = Reranker(scorers[v]).rerank(pool['query'], hits)
                        elapsed = (perf_counter()-start)*1000
                        fallbacks = {h.metadata['rerank'].get('fallback') for h in result} - {None}
                        scores = [] if fallbacks else [None]*len(hits)
                        if not fallbacks:
                            for hit in result: scores[hit.metadata['rerank']['input_rank']-1] = hit.score
                        by_source = {h.source: d for h, d in zip(hits, pool['candidate_document_ids'])}
                        ranking = [by_source[h.source] for h in result] + pool['hybrid_ranking'][20:]
                        for rank, d in enumerate(inputs):
                            d.update(original_hybrid_rank=rank+1, score=scores[rank] if scores else None)
                        emit(streams[v], {'dataset': ds, 'qid': pool['qid'], 'inputs': inputs,
                            'scores': scores, 'ranking': ranking, 'elapsed_ms': elapsed,
                            'fallback': next(iter(fallbacks)) if fallbacks else None,
                            'result_provenance': [h.metadata['rerank'] for h in result]})
                    if i % 10 == 0 or i == len(pools)-1:
                        print('B3/B4', ds, i+1, '/', len(pools), flush=True)
            finally:
                for stream in streams.values(): stream.close()
        check_source()
        if model_files()[1] != protocol['model_files']:
            raise ValueError('Model file drift.')
        for value in meta.values():
            if environment() != value['environment']: raise ValueError('Software environment drift.')
            value['status'] = 'completed'
    except BaseException as error:
        for value in meta.values():
            value.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        for v, folder in folders.items():
            meta[v]['finished_at'] = now(); write_json(folder/'experiment.json', meta[v])
            write_json(folder/'checksums.json', {p.name: digest(p) for p in folder.iterdir()
                                                if p.is_file() and p.name != 'checksums.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'baseline', 'instrument', 'register', 'execute'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    {'freeze': freeze, 'baseline': baseline, 'instrument': instrument,
     'register': register, 'execute': execute_allocations}[args.action](args.output.resolve())


if __name__ == '__main__':
    main()

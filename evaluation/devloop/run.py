"""Run the product agent (or the frozen evaluation adapter) on the development set and score it.

This is a development instrument: about 150 scenarios, minutes not days, no
judge, no statistical claims. Metrics are evidence delivery, completeness and
cost. Labels are read only after inference, from labels.json.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import httpx
import numpy as np
from ollama import ChatResponse

from arkb.agent.observation import AgentObserver
from arkb.config import RuntimeConfig
from arkb.evaluation.deadline import evaluation_deadline
from arkb.evaluation.external import digest, write_json
from arkb.evaluation.v2 import load_dataset, evidence_scores
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.runtime import Runtime
from evaluation.agentic_tools.analyze import first_positive_discovery
from evaluation.agentic_tools.contract import (ARM_BY_ID, BUDGET, OPTIONS, controlled_agent, evidence_sets, fixed_rag,
                                               new_observer)
from evaluation.agentic_tools.prepare import utc
from evaluation.agentic_tools.scoring import canonical, costs_and_behavior, evidence_coverage, musique_row
from evaluation.agentic_tools.transport import LocalChatClient, model_identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPTIONS = dict(OPTIONS)
PRIMARY = {'v2': 'evidence_coverage_delivered', 'exact-nfcorpus': 'completeness_cited',
           'recall-nfcorpus': 'positive_recall_delivered', 'recall-fiqa': 'positive_recall_delivered',
           'musique': 'answer_f1', 'long-browsecomp': 'positive_recall_delivered'}


class DevChatClient:
    """Direct Ollama transport with explicit model/options/thinking; journals nothing by default."""

    def __init__(self, model, *, options, think):
        self.model, self.options, self.think = model, dict(options), think
        self.http = httpx.Client(base_url='http://127.0.0.1:11434', timeout=httpx.Timeout(330, connect=10))
        self.identity = model_identity(self.http, model)

    def chat(self, **request):
        request = dict(request, model=self.model, think=self.think, truncate=False, shift=False,
                       options={**(request.get('options') or {}), **self.options})
        response = self.http.post('/api/chat', json=request)
        response.raise_for_status()
        value = response.json()
        if 'error' in value:
            raise ValueError('Ollama response error: ' + value['error'])
        return ChatResponse.model_validate(value)

    def close(self):
        self.http.close()


def stage_observations(report):
    """Bound evidence per stage; the shape the v2 span scorer accepts."""
    refs = report.get('evidence_references') or {}
    stages = {'returned': [], 'delivered': [], 'submitted': []}
    for event in report.get('tools') or []:
        raw = event.get('raw_result') or {}
        hits = [raw['result']] if 'result' in raw else raw.get('results', [])
        allowed = set(event.get('delivered_refs', [h.get('ref') for h in hits if isinstance(h, dict)]))
        for hit in hits:
            bound = refs.get(hit.get('ref')) if isinstance(hit, dict) else None
            if not bound:
                continue
            observation = {k: bound.get(k) for k in ('source', 'document_revision', 'start_char', 'end_char', 'content')}
            stages['returned'].append(observation)
            if event.get('delivered_to_conversation') and hit['ref'] in allowed:
                stages['delivered'].append(observation)
                if event.get('submitted_to_model'):
                    stages['submitted'].append(observation)
    return stages


def cited_sources(row):
    final = (row.get('result') or {}).get('final') or {}
    return sorted({c['source'] for c in final.get('citations') or [] if isinstance(c, dict) and c.get('source')})


def executed_tools(row):
    report = (row.get('result') or {}).get('observation') or {}
    return [t for t in report.get('tools') or [] if t.get('executed')]


def match_limit_hits(row):
    hits = 0
    for event in executed_tools(row):
        if event['name'] == 'match' and event.get('status') == 'success':
            limit = (event.get('arguments') or {}).get('limit', 5)
            if len((event.get('raw_result') or {}).get('results', [])) >= limit:
                hits += 1
    return hits


def score_row(row, scenario, labels, resources):
    """Offline scoring from the saved record; never influences inference."""
    slice_name = scenario['slice']
    # Shared scorers expect a registered schedule; the product agent behaves like A-All.
    row = {**row, 'schedule': {'arm': 'A-All', 'id': scenario.get('source_id', scenario['id'])}}
    report = (row.get('result') or {}).get('observation') or {}
    final = canonical(row)
    status = final.status if final else 'error'
    stages = stage_observations(report)
    sources = {stage: sorted({o['source'] for o in obs}) for stage, obs in stages.items()}
    cited = cited_sources(row)
    tools = executed_tools(row)
    scored = {'final_status': status, 'stop_reason': (row.get('result') or {}).get('stop_reason', 'error'),
              'error': row.get('error'), 'costs': costs_and_behavior(row),
              'tool_names': [t['name'] for t in tools], 'sources': sources, 'cited_sources': cited}
    if slice_name == 'v2':
        dataset = resources['v2']
        case = next(c for c in dataset.cases if c['id'] == scenario['id'])
        expected = {r['source'] for r in dataset.qrels if r['query_id'] == case['id'] and r['grade'] >= 2}
        for stage, obs in stages.items():
            e = evidence_scores(case, obs, dataset)
            scored[f'evidence_coverage_{stage}'] = e['evidence_coverage']
            scored[f'critical_covered_{stage}'] = e['all_critical_facets_covered']
        scored['evidence_coverage_cited'] = None
        scored['expected_sources'] = sorted(expected)
        for stage, found in list(sources.items()) + [('cited', cited)]:
            scored[f'source_recall_{stage}'] = len(expected & set(found)) / len(expected) if expected else None
        task = scenario['task_type']
        scored['behavior'] = {
            'no_retrieval_respected': not tools if task == 'no_retrieval' else None,
            'read_only_respected': all(t['name'] in ('read', 'finish') for t in tools) if task == 'direct_read' else None,
            'abstained': status == 'insufficient_evidence' if task == 'evidence_gap' else None,
            'answered': status in ('answered', 'partial')}
    elif slice_name.startswith('exact-'):
        expected = set(labels['expected_sources'])
        for stage, found in list(sources.items()) + [('cited', cited)]:
            scored[f'completeness_{stage}'] = len(expected & set(found)) / len(expected) if expected else None
            scored[f'spurious_{stage}'] = len(set(found) - expected)
        scored['expected_count'] = len(expected)
        scored['match_calls'] = sum(t['name'] == 'match' for t in tools)
        scored['match_limit_hits'] = match_limit_hits(row)
        scored['complete_and_exact_cited'] = set(cited) == expected
    elif slice_name.startswith('recall-') or slice_name == 'long-browsecomp':
        dataset = scenario['scope']
        label = {'qrels': {scenario['source_id']: labels['qrels']}, 'source_map': resources['source_maps'][dataset]}
        if 'gold_qrels' in labels:
            label['gold_qrels'] = {scenario['source_id']: labels['gold_qrels']}
        row_for = row
        coverage = evidence_coverage(row_for, label)
        for stage in ('returned', 'delivered', 'submitted'):
            scored[f'positive_recall_{stage}'] = coverage[f'qrels_{stage}']['positive_recall']
            scored[f'unjudged_{stage}'] = coverage[f'qrels_{stage}']['unjudged_sources']
            if f'gold_qrels_{stage}' in coverage:
                scored[f'gold_recall_{stage}'] = coverage[f'gold_qrels_{stage}']['positive_recall']
        scored['budget_stop_reason'] = report.get('budget_stop_reason')
        scored['delivered_evidence_tokens'] = (report.get('evidence') or {}).get('delivered_tokens')
        scored['withheld_reads'] = sum(1 for t in tools if t['name'] == 'read' and not t.get('delivered_to_conversation'))
        scored['first_positive'] = first_positive_discovery(row_for, label)
        scored['answered'] = status in ('answered', 'partial')
    elif slice_name == 'musique':
        scored['musique'] = musique_row(row, labels['gold'])
        m = scored['musique']
        scored['answer_f1'], scored['answer_em'], scored['support_f1'] = m['answer_f1'], m['answer_em'], m['support_f1']
        scored['answerability_correct'] = m['answerability_correct']
    return scored


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize(records):
    by_slice = defaultdict(list)
    for r in records:
        by_slice[r['scenario']['slice']].append(r)
    summary = {}
    for name, rows in sorted(by_slice.items()):
        s = [r['scores'] for r in rows]
        block = {'scenarios': len(rows), 'errors': sum(bool(x['error']) for x in s),
                 'final_statuses': dict(Counter(x['final_status'] for x in s)),
                 'stop_reasons': dict(Counter(x['stop_reason'] for x in s)),
                 'elapsed_seconds_mean': mean([x['costs']['elapsed_ms'] / 1000 for x in s]),
                 'elapsed_seconds_p95': float(np.quantile([x['costs']['elapsed_ms'] / 1000 for x in s], .95)),
                 'model_requests_mean': mean([x['costs']['model_requests'] for x in s]),
                 'tool_calls_mean': mean([len(x['tool_names']) for x in s]),
                 'tools_executed': dict(sum((Counter(x['tool_names']) for x in s), Counter())),
                 'prompt_tokens_mean': mean([((x['costs']['usage'] or {}).get('prompt_eval_count') or {}).get('known_total') for x in s]),
                 'eval_tokens_mean': mean([((x['costs']['usage'] or {}).get('eval_count') or {}).get('known_total') for x in s]),
                 'delivered_evidence_tokens_mean': mean([(x['costs']['evidence_tokens'] or {}).get('delivered_tokens') for x in s])}
        metrics = [k for k in s[0] if k.startswith(('evidence_coverage_', 'critical_covered_', 'source_recall_', 'completeness_',
                                                     'positive_recall_', 'gold_recall_', 'answer_', 'support_f1', 'answerability_correct',
                                                     'complete_and_exact_cited', 'delivered_evidence_tokens', 'withheld_reads'))]
        if 'budget_stop_reason' in s[0]:
            block['budget_stop_reasons'] = dict(Counter(x['budget_stop_reason'] for x in s))
        for metric in metrics:
            block[metric] = mean([x.get(metric) for x in s])
        if name == 'v2':
            by_task = defaultdict(list)
            for r in rows:
                by_task[r['scenario']['task_type']].append(r['scores'])
            block['by_task_type'] = {task: {'scenarios': len(v), 'evidence_coverage_delivered': mean([x['evidence_coverage_delivered'] for x in v]),
                                            'source_recall_delivered': mean([x['source_recall_delivered'] for x in v]),
                                            'answered': mean([x['behavior']['answered'] for x in v]),
                                            'no_retrieval_respected': mean([x['behavior']['no_retrieval_respected'] for x in v]),
                                            'read_only_respected': mean([x['behavior']['read_only_respected'] for x in v]),
                                            'abstained': mean([x['behavior']['abstained'] for x in v])}
                                     for task, v in sorted(by_task.items())}
        if name.startswith('exact-'):
            block['match_limit_hits_total'] = sum(x['match_limit_hits'] for x in s)
            block['by_stratum'] = {}
            strata = defaultdict(list)
            for r in rows:
                strata[r['scenario']['stratum']].append(r['scores'])
            for stratum, v in sorted(strata.items()):
                block['by_stratum'][stratum] = {'scenarios': len(v), 'completeness_cited': mean([x['completeness_cited'] for x in v]),
                                                'completeness_returned': mean([x['completeness_returned'] for x in v]),
                                                'complete_and_exact_cited': mean([x['complete_and_exact_cited'] for x in v])}
        if name == 'musique':
            pairs = defaultdict(dict)
            for r in rows:
                pairs[r['scenario']['pair_id']][r['scenario']['variant']] = r['scores']['musique']
            block['pair_sufficiency'] = mean([float(all(v['answerability_correct'] for v in p.values())) for p in pairs.values() if len(p) == 2])
        summary[name] = block
    return summary


def markdown(summary, meta):
    lines = [f"# Dev loop run `{meta['label']}`", '',
             f"adapter={meta['adapter']} model={meta['model']} think={meta['think']} git={meta['git_head'][:10]}{'+dirty' if meta['dirty'] else ''} scenarios={meta['scenarios']} wall={meta['wall_seconds']:.0f}s", '',
             '| slice | n | primary | value | errors | elapsed mean s | model req | tool calls |', '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for name, block in summary.items():
        primary = PRIMARY.get(name, '')
        value = block.get(primary)
        lines.append(f"| {name} | {block['scenarios']} | {primary} | {value if value is None else f'{value:.3f}'} | {block['errors']} | "
                     f"{block['elapsed_seconds_mean']:.1f} | {block['model_requests_mean']:.2f} | {block['tool_calls_mean']:.2f} |")
    for name, block in summary.items():
        lines += ['', f'## {name}', '']
        for key, value in block.items():
            if isinstance(value, dict) and key in ('by_task_type', 'by_stratum'):
                lines.append(f'- {key}:')
                for sub, inner in value.items():
                    lines.append(f"  - {sub}: " + ', '.join(f"{k}={v if not isinstance(v, float) else round(v, 3)}" for k, v in inner.items()))
            elif isinstance(value, float):
                lines.append(f'- {key}: {value:.3f}')
            else:
                lines.append(f'- {key}: {value}')
    return '\n'.join(lines) + '\n'


def source_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'evaluation/devloop', 'evaluation/agentic_tools'], cwd=ROOT, text=True).strip()
    files = {str(p.relative_to(ROOT)): digest(p) for base in (ROOT / 'src', ROOT / 'evaluation/devloop') for p in sorted(base.rglob('*.py'))}
    return {'git_head': head, 'dirty': bool(dirty), 'dirty_files': dirty.splitlines(),
            'source_sha256': digest_of(files)}


def digest_of(mapping):
    import hashlib
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def load_resources(devset):
    return {'v2': load_dataset(ROOT / 'evaluation/data/v2/pilot', notes_dir=ROOT / 'evaluation/data/v2/pilot/corpus', allow_provisional=True),
            'source_maps': {p.stem: json.loads(p.read_text())['source_map'] for p in sorted((devset / 'scoring').glob('*.json'))}}


def run(devset, output, *, adapter, model, think, options, slices, limit, label, max_turns=8, resources=None, resume=False):
    resources = resources or load_resources(devset)
    scenarios = json.loads((devset / 'scenarios.json').read_text())
    labels = json.loads((devset / 'labels.json').read_text())
    scopes = json.loads((devset / 'scopes.json').read_text())
    if slices:
        scenarios = [s for s in scenarios if s['slice'] in slices]
    else:
        scenarios = [s for s in scenarios if not s.get('optional')]
    if limit:
        per = defaultdict(int)
        kept = []
        for s in scenarios:
            if per[s['slice']] < limit:
                kept.append(s)
                per[s['slice']] += 1
        scenarios = kept
    output.mkdir(parents=True, exist_ok=resume)
    results_path = output / 'results.jsonl'
    records = []
    if resume and results_path.exists():
        records = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        done = {r['scenario']['id'] for r in records}
        scenarios = [s for s in scenarios if s['id'] not in done]
        meta = json.loads((output / 'run.json').read_text())
        meta.update(resumed_at=utc(), resumed_from=len(records))
    else:
        meta = {'label': label, 'adapter': adapter, 'model': model, 'think': think, 'options': options, 'budget': asdict(BUDGET),
                'max_turns': max_turns, 'devset_manifest': json.loads((devset / 'manifest.json').read_text()),
                'started_at': utc(), 'scenarios': len(scenarios), **source_state()}
    write_json(output / 'run.json', meta)
    started = perf_counter()
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(),
                               qdrant_url='http://127.0.0.1:6340')) as runtime, ExitStack() as stack:
        tokenizer = runtime.tokenizer()
        from arkb.knowledge.embeddings import tokenizer_fingerprint
        counter_id = 'reference-text:' + tokenizer_fingerprint(tokenizer)
        counter = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
        if adapter == 'product':
            client = DevChatClient(model, options=options, think=think)
        else:
            client = LocalChatClient(model_identity(httpx.Client(base_url='http://127.0.0.1:11434', timeout=30), model))
        meta['chat_model'] = client.identity
        write_json(output / 'run.json', meta)
        setup_path = output / 'setup-costs.json'
        active, tools, setup = None, None, (json.loads(setup_path.read_text()) if resume and setup_path.exists() else [])
        for index, scenario in enumerate(scenarios):
            scope_name = scenario['scope']
            if scope_name != active:
                tools = None
                stack.close()
                start = perf_counter()
                scope = scopes.get(scope_name) or {'corpus': scenario['corpus'], 'sqlite': scenario['sqlite'], 'vault_id': scenario['vault_id'], 'prepare_exact': True}
                storage = stack.enter_context(SQLiteStorage(Path(scope['sqlite']), read_only=True))
                manifest = storage.active_manifest(scope['vault_id'])
                if scope.get('index_manifest') and asdict(manifest) != scope['index_manifest']:
                    raise ValueError('Index manifest differs from the registered scope: ' + scope_name)
                engine = runtime.retrieval_engine(storage, manifest, modes=('bm25', 'semantic'), exact=True)
                tools = runtime.agent_tools(engine=engine, directory=Path(scope['corpus']), vault_id=scope['vault_id'],
                                            rerank=False, prepare_exact=scope.get('prepare_exact', True))
                stack.callback(tools._exact.close)
                setup.append({'scope': scope_name, 'elapsed_ms': (perf_counter() - start) * 1000, 'index_id': manifest.index_version})
                write_json(output / 'setup-costs.json', setup)
                active = scope_name
            # The product loop sets temperature only; DevChatClient supplies context/output limits.
            observer = (AgentObserver(budget=BUDGET, counter=counter, counter_identity=counter_id)
                        if adapter == 'product' else new_observer(counter, counter_id))
            start = perf_counter()
            result, error = None, None
            try:
                with evaluation_deadline(360):
                    if adapter == 'product':
                        result = runtime.run_agent(scenario['query'], tools=tools, model=model, max_turns=max_turns,
                                                   think=think, client=client, observer=observer)
                    else:
                        arm = ARM_BY_ID[adapter.split(':', 1)[1]]
                        fn = fixed_rag if arm.fixed else controlled_agent
                        result = fn(scenario['query'], tools=tools, arm=arm, client=client, observer=observer)
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
            elapsed_ms = (perf_counter() - start) * 1000
            row = {'scenario': scenario, 'elapsed_ms': elapsed_ms, 'error': error,
                   'result': asdict(result) if result else None, 'completed_at': utc()}
            if result:
                row['evidence_sources'] = evidence_sets(result.observation)
            row['scores'] = score_row(row, scenario, labels[scenario['id']], resources)
            records.append(row)
            with results_path.open('a') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            print(json.dumps({'done': index + 1, 'total': len(scenarios), 'slice': scenario['slice'], 'id': scenario['id'],
                              'status': row['scores']['final_status'], 'elapsed_s': round(elapsed_ms / 1000, 1),
                              'primary': row['scores'].get(PRIMARY.get(scenario['slice'], ''))}), flush=True)
        client.close()
    summary = summarize(records)
    meta.update(completed_at=utc(), wall_seconds=meta.get('wall_seconds', 0) + perf_counter() - started, setup=setup)
    write_json(output / 'run.json', meta)
    write_json(output / 'summary.json', summary)
    (output / 'summary.md').write_text(markdown(summary, meta))
    return summary


def rescore(output, devset):
    """Recompute scores and summary from saved records with the current scorer."""
    resources = load_resources(devset)
    labels = json.loads((devset / 'labels.json').read_text())
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines() if line.strip()]
    for row in rows:
        row['scores'] = score_row(row, row['scenario'], labels[row['scenario']['id']], resources)
    (output / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    meta = json.loads((output / 'run.json').read_text())
    summary = summarize(rows)
    meta.update(rescored_at=utc(), scorer_source_sha256=digest(Path(__file__)))
    write_json(output / 'run.json', meta)
    write_json(output / 'summary.json', summary)
    (output / 'summary.md').write_text(markdown(summary, meta))
    return summary


def compare(a, b, destination=None):
    """Paired comparison of two runs on shared scenario IDs; descriptive only."""
    def load(path):
        rows = [json.loads(l) for l in (path / 'results.jsonl').read_text().splitlines()]
        return json.loads((path / 'run.json').read_text()), {r['scenario']['id']: r for r in rows}
    meta_a, rows_a = load(a)
    meta_b, rows_b = load(b)
    shared = sorted(set(rows_a) & set(rows_b))
    lines = [f"# Dev loop comparison: `{meta_a['label']}` vs `{meta_b['label']}`", '',
             f"A: {meta_a['adapter']} {meta_a['model']} think={meta_a['think']} git={meta_a['git_head'][:10]}{'+dirty' if meta_a['dirty'] else ''}",
             f"B: {meta_b['adapter']} {meta_b['model']} think={meta_b['think']} git={meta_b['git_head'][:10]}{'+dirty' if meta_b['dirty'] else ''}",
             f'shared scenarios: {len(shared)}', '',
             '| slice | metric | A | B | delta | wins | ties | losses |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    result = {}
    by_slice = defaultdict(list)
    for sid in shared:
        by_slice[rows_a[sid]['scenario']['slice']].append(sid)
    for name, ids in sorted(by_slice.items()):
        metrics = [PRIMARY.get(name, '')] + [m for m in ('evidence_coverage_returned', 'source_recall_cited', 'completeness_returned',
                                                            'positive_recall_returned', 'answerability_correct', 'support_f1') if m in rows_a[ids[0]]['scores']]
        for metric in dict.fromkeys(m for m in metrics if m):
            pairs = [(rows_a[i]['scores'].get(metric), rows_b[i]['scores'].get(metric)) for i in ids]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                continue
            va, vb = float(np.mean([x for x, _ in pairs])), float(np.mean([y for _, y in pairs]))
            wins = sum(y > x + 1e-12 for x, y in pairs)
            losses = sum(y < x - 1e-12 for x, y in pairs)
            result[f'{name}:{metric}'] = {'a': va, 'b': vb, 'delta': vb - va, 'wins': wins, 'ties': len(pairs) - wins - losses, 'losses': losses, 'n': len(pairs)}
            lines.append(f'| {name} | {metric} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} | {wins} | {len(pairs) - wins - losses} | {losses} |')
        ea = float(np.mean([rows_a[i]['scores']['costs']['elapsed_ms'] / 1000 for i in ids]))
        eb = float(np.mean([rows_b[i]['scores']['costs']['elapsed_ms'] / 1000 for i in ids]))
        lines.append(f'| {name} | elapsed_seconds_mean | {ea:.1f} | {eb:.1f} | {eb - ea:+.1f} | | | |')
        result[f'{name}:elapsed_seconds_mean'] = {'a': ea, 'b': eb, 'delta': eb - ea}
    text = '\n'.join(lines) + '\n'
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    return result, text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--devset', type=Path, default=ROOT / 'evaluation/devloop/devset-v1')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--label', required=True)
    r.add_argument('--adapter', default='product', help="product | contract:<ARM> (e.g. contract:A-All)")
    r.add_argument('--model', default='qwen3.5:4b')
    r.add_argument('--think', action='store_true')
    r.add_argument('--num-ctx', type=int, default=DEFAULT_OPTIONS['num_ctx'])
    r.add_argument('--num-predict', type=int, default=DEFAULT_OPTIONS['num_predict'])
    r.add_argument('--slices', default='')
    r.add_argument('--limit', type=int, default=0)
    r.add_argument('--max-turns', type=int, default=8)
    r.add_argument('--resume', action='store_true', help='continue an interrupted run in the same output directory')
    rs = sub.add_parser('rescore')
    rs.add_argument('--devset', type=Path, default=ROOT / 'evaluation/devloop/devset-v1')
    rs.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('a', type=Path)
    c.add_argument('b', type=Path)
    c.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.command == 'run':
        options = {'temperature': 0, 'num_ctx': args.num_ctx, 'num_predict': args.num_predict}
        summary = run(args.devset.resolve(), args.output.resolve(), adapter=args.adapter, model=args.model, think=args.think,
                      options=options, slices=[s for s in args.slices.split(',') if s], limit=args.limit, label=args.label,
                      max_turns=args.max_turns, resume=args.resume)
        print(json.dumps({name: {k: block.get(k) for k in ('scenarios', 'errors', PRIMARY.get(name, ''))} for name, block in summary.items()}, indent=2))
    elif args.command == 'rescore':
        summary = rescore(args.output.resolve(), args.devset.resolve())
        print(json.dumps({name: {k: block.get(k) for k in ('scenarios', 'errors', PRIMARY.get(name, ''))} for name, block in summary.items()}, indent=2))
    else:
        result, text = compare(args.a.resolve(), args.b.resolve(), args.output.resolve() if args.output else None)
        print(text)


if __name__ == '__main__':
    main()

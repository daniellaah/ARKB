"""Run the product agent on the development set and score it.

A development instrument: 84 core questions in about ten minutes, optional
slices when the external volume is mounted, no judge, no statistical claims.
Labels are read after inference, by score.py, never here.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from time import perf_counter

from arkb.agent.observation import AgentBudget, AgentObserver
from arkb.config import RuntimeConfig
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.runtime import Runtime

from .common import ROOT, deadline, digest, digest_text, read_json, resolve, utc, write_json
from .score import PRIMARY, load_resources, markdown, score_row, summarize
from .transport import OllamaClient

DEVSET = ROOT / 'evaluation/devset'
QDRANT_URL = 'http://127.0.0.1:6340'
DEFAULT_OPTIONS = {'temperature': 0, 'num_ctx': 32768, 'num_predict': 4096}
DEFAULT_BUDGET = AgentBudget(max_tool_calls=12, max_query_calls=10, max_read_calls=6,
                             max_evidence_tokens=8000, max_elapsed_ms=300000)
SCENARIO_DEADLINE_SECONDS = 360


def source_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'evaluation'], cwd=ROOT, text=True).strip()
    dirty_files = [line for line in dirty.splitlines() if not line.endswith(('.md', '.json'))]
    files = {str(p.relative_to(ROOT)): digest(p) for base in (ROOT / 'src', ROOT / 'evaluation')
             for p in sorted(base.rglob('*.py')) if 'results' not in p.parts}
    return {'git_head': head, 'dirty': bool(dirty_files), 'dirty_files': dirty_files,
            'source_sha256': digest_text(json.dumps(files, sort_keys=True))}


def scope_available(scope):
    return all(resolve(scope[k]).exists() for k in ('corpus', 'sqlite'))


def select_scenarios(scenarios, scopes, slices, limit):
    """`core` is every non-optional slice; `all` adds optional slices whose inputs are present; names select explicitly."""
    skipped = {}
    if slices == ['core']:
        chosen = [s for s in scenarios if not s.get('optional')]
    else:
        wanted = None if slices == ['all'] else set(slices)
        chosen = []
        for s in scenarios:
            if wanted is not None and s['slice'] not in wanted:
                continue
            scope = scopes.get(s['scope']) or {'corpus': s['corpus'], 'sqlite': s['sqlite']}
            if s.get('optional') and not scope_available(scope):
                skipped[s['slice']] = 'inputs not present'
                continue
            chosen.append(s)
    if limit:
        per = defaultdict(int)
        kept = []
        for s in chosen:
            if per[s['slice']] < limit:
                kept.append(s)
                per[s['slice']] += 1
        chosen = kept
    return chosen, skipped


def run(output, *, label, model, think, options, budget=DEFAULT_BUDGET, slices=('core',), limit=0, max_turns=8,
        devset=DEVSET, resume=False):
    scenarios = read_json(devset / 'scenarios.json')
    labels = read_json(devset / 'labels.json')
    scopes = read_json(devset / 'scopes.json')
    scenarios, skipped = select_scenarios(scenarios, scopes, list(slices), limit)
    output.mkdir(parents=True, exist_ok=resume)
    results_path = output / 'results.jsonl'
    records = []
    if resume and results_path.exists():
        records = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        done = {r['scenario']['id'] for r in records}
        scenarios = [s for s in scenarios if s['id'] not in done]
        meta = read_json(output / 'run.json')
        meta.update(resumed_at=utc(), resumed_from=len(records))
    else:
        meta = {'label': label, 'model': model, 'think': think, 'options': options, 'budget': asdict(budget),
                'max_turns': max_turns, 'slices': list(slices), 'skipped_slices': skipped,
                'devset_manifest': read_json(devset / 'manifest.json'), 'started_at': utc(),
                'scenarios': len(scenarios), **source_state()}
    write_json(output / 'run.json', meta)
    resources = load_resources(devset)
    started = perf_counter()
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(),
                               qdrant_url=QDRANT_URL)) as runtime, ExitStack() as stack:
        tokenizer = runtime.tokenizer()
        from arkb.knowledge.embeddings import tokenizer_fingerprint
        counter_id = 'reference-text:' + tokenizer_fingerprint(tokenizer)
        counter = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
        client = OllamaClient(model, options=options, think=think)
        meta['chat_model'] = client.identity
        write_json(output / 'run.json', meta)
        setup_path = output / 'setup-costs.json'
        active, tools, setup = None, None, (read_json(setup_path) if resume and setup_path.exists() else [])
        for index, scenario in enumerate(scenarios):
            scope_name = scenario['scope']
            if scope_name != active:
                tools = None
                stack.close()
                start = perf_counter()
                scope = scopes.get(scope_name) or {'corpus': scenario['corpus'], 'sqlite': scenario['sqlite'],
                                                   'vault_id': scenario['vault_id'], 'prepare_exact': True}
                storage = stack.enter_context(SQLiteStorage(resolve(scope['sqlite']), read_only=True))
                manifest = storage.active_manifest(scope['vault_id'])
                if scope.get('index_manifest') and asdict(manifest) != scope['index_manifest']:
                    raise ValueError('Index manifest differs from the devset scope: ' + scope_name)
                engine = runtime.retrieval_engine(storage, manifest, modes=('bm25', 'semantic'), exact=True)
                tools = runtime.agent_tools(engine=engine, directory=resolve(scope['corpus']), vault_id=scope['vault_id'],
                                            rerank=False, prepare_exact=scope.get('prepare_exact', True))
                stack.callback(tools._exact.close)
                setup.append({'scope': scope_name, 'elapsed_ms': (perf_counter() - start) * 1000, 'index_id': manifest.index_version})
                write_json(setup_path, setup)
                active = scope_name
            # The product loop sets temperature only; the client supplies context and output limits.
            observer = AgentObserver(budget=budget, counter=counter, counter_identity=counter_id)
            start = perf_counter()
            result, error = None, None
            try:
                with deadline(SCENARIO_DEADLINE_SECONDS):
                    result = runtime.run_agent(scenario['query'], tools=tools, model=model, max_turns=max_turns,
                                               think=think, client=client, observer=observer)
            except Exception as exc:
                error = {'type': type(exc).__name__, 'message': str(exc)}
            elapsed_ms = (perf_counter() - start) * 1000
            row = {'scenario': scenario, 'elapsed_ms': elapsed_ms, 'error': error,
                   'result': asdict(result) if result else None, 'completed_at': utc()}
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


def rescore(output, devset=DEVSET):
    """Recompute scores and summary from saved records with the current scorer; records stay otherwise untouched."""
    resources = load_resources(devset)
    labels = read_json(devset / 'labels.json')
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines() if line.strip()]
    for row in rows:
        row['scores'] = score_row(row, row['scenario'], labels[row['scenario']['id']], resources)
    (output / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    meta = read_json(output / 'run.json')
    summary = summarize(rows)
    meta.update(rescored_at=utc(), scorer_sha256=digest(Path(__file__).with_name('score.py')))
    write_json(output / 'run.json', meta)
    write_json(output / 'summary.json', summary)
    (output / 'summary.md').write_text(markdown(summary, meta))
    return summary


def brief(summary):
    return {name: {k: block.get(k) for k in ('scenarios', 'errors', PRIMARY.get(name, ''))} for name, block in summary.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='run the product agent on the devset')
    r.add_argument('--label', required=True)
    r.add_argument('--output', type=Path, help='defaults to evaluation/results/<label>')
    r.add_argument('--devset', type=Path, default=DEVSET)
    r.add_argument('--model', default='qwen3.5:9b')
    r.add_argument('--think', action='store_true')
    r.add_argument('--num-ctx', type=int, default=DEFAULT_OPTIONS['num_ctx'])
    r.add_argument('--num-predict', type=int, default=DEFAULT_OPTIONS['num_predict'])
    r.add_argument('--max-turns', type=int, default=8)
    r.add_argument('--slices', default='core', help='core (default) | all | comma-separated slice names')
    r.add_argument('--limit', type=int, default=0, help='scenarios per slice, for quick checks')
    r.add_argument('--max-tool-calls', type=int, default=DEFAULT_BUDGET.max_tool_calls)
    r.add_argument('--max-query-calls', type=int, default=DEFAULT_BUDGET.max_query_calls)
    r.add_argument('--max-read-calls', type=int, default=DEFAULT_BUDGET.max_read_calls)
    r.add_argument('--max-evidence-tokens', type=int, default=DEFAULT_BUDGET.max_evidence_tokens)
    r.add_argument('--max-seconds', type=int, default=DEFAULT_BUDGET.max_elapsed_ms // 1000)
    r.add_argument('--resume', action='store_true', help='continue an interrupted run in the same output directory')
    rs = sub.add_parser('rescore', help='recompute scores of a saved run with the current scorer')
    rs.add_argument('output', type=Path)
    rs.add_argument('--devset', type=Path, default=DEVSET)
    args = p.parse_args()
    if args.command == 'run':
        options = {'temperature': 0, 'num_ctx': args.num_ctx, 'num_predict': args.num_predict}
        budget = AgentBudget(max_tool_calls=args.max_tool_calls, max_query_calls=args.max_query_calls,
                             max_read_calls=args.max_read_calls, max_evidence_tokens=args.max_evidence_tokens,
                             max_elapsed_ms=args.max_seconds * 1000)
        output = (args.output or ROOT / 'evaluation/results' / args.label).resolve()
        summary = run(output, label=args.label, model=args.model, think=args.think, options=options, budget=budget,
                      slices=[s for s in args.slices.split(',') if s], limit=args.limit, max_turns=args.max_turns,
                      devset=args.devset.resolve(), resume=args.resume)
    else:
        summary = rescore(args.output.resolve(), args.devset.resolve())
    print(json.dumps(brief(summary), indent=2))


if __name__ == '__main__':
    main()

"""Run the teacher over the vault questions, several samples each, in parallel.

`evaluation/eval.py` runs one question at a time, which is what an evaluation
should do: a shared index, one wall clock, one ordering. Generating training
data has no such obligation, and the teacher is a network call that spends
most of its time waiting. So this runs W processes, not threads: a `Runtime`
owns SQLite connections, a warmed literal-text cache and a Qdrant client, none
of which is safe to share, and each worker therefore builds its own.

Everything a worker runs inside that composition is the product's: the same
agent loop, the same tool session, the same budgets, and `evaluation.eval.score`
for the scores, so a trajectory kept here is one the evaluation would recognise.
The two departures are deliberate and named: the model is sampled rather than
greedy (four identical trajectories teach nothing), and the corpus is the real
vault rather than the 58-note evaluation set, which must never be trained on.

Spending is estimated from the transport's own usage rows, which count
DeepSeek's cache hits at the cache rate. Each worker holds a share of the
budget and stops when it has spent it, so the run cannot overrun while nobody
is watching.

    python -m training.rollout --questions training/data/questions-pilot.json \\
        --out training/data/rollouts-pilot --samples 4 --workers 8 --max-cost 3.0
"""

import argparse
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
from time import perf_counter, strftime

from arkb.config import DEFAULT_RETRIEVAL_MODE

from training import chat_format

ROOT = Path(__file__).resolve().parents[1]
VAULT = Path('/Users/daboluo/ObsidianVault/MyObsidian')
INDEX = ROOT / '.arkb/obsidian.sqlite'
VAULT_ID = 'obsidian'
QDRANT_URL = 'http://127.0.0.1:6340'


def worker(shard, tasks, settings, out_dir):
    """One process: its own runtime, its own transport, its own share of the budget."""
    from arkb.agent import tools as agent_tools_module
    from arkb.agent.loop import SYSTEM_INSTRUCTION, run_agent
    from arkb.agent.observation import AgentObserver
    from arkb.agent.transports import ChatUsage, estimate_cost, make_client
    from arkb.config import RuntimeConfig, load_env_file
    from arkb.knowledge.embeddings import tokenizer_fingerprint
    from arkb.knowledge.links import LinkGraph
    from arkb.knowledge.sqlite import SQLiteStorage
    from arkb.runtime import Runtime
    from evaluation.eval import BUDGET, QUESTION_DEADLINE_SECONDS, deadline, score
    from dataclasses import asdict
    from training.transport import MlxServerClient, SampledDeepSeekClient

    load_env_file(ROOT / '.env')
    # An A/B on retrieval policy changes what the agent is told, not what the
    # agent is. The query hint belongs to the tool schema, where the model
    # reads it while building the call and where this project puts retrieval
    # policy; it is patched in the worker so a candidate can be measured
    # before it is committed to the product.
    if settings['query_hint']:
        for definition in agent_tools_module.TOOL_DEFINITIONS:
            if definition['name'] == 'search':
                definition['parameters']['properties']['query']['description'] = settings['query_hint']
    instruction = (Path(settings['system_instruction']).read_text().strip()
                   if settings['system_instruction'] else SYSTEM_INSTRUCTION)
    rows_path = out_dir / f'results-{shard}.jsonl'
    # The Ollama options the harness sets (num_ctx, num_predict) belong to a
    # local server; this transport carries its own max_tokens, and the sampling
    # temperature is the client's, because the loop pins zero in every request.
    spent, stopped = 0.0, None
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(),
                               qdrant_url=QDRANT_URL)) as runtime:
        tokenizer = runtime.tokenizer()
        counter_id = 'reference-text:' + tokenizer_fingerprint(tokenizer)
        # Three transports, one composition: the paid teacher, a local model
        # that rehearses the whole thing for nothing, and the served student,
        # which is how the same vault questions measure a fine-tune in domain.
        if settings['base_url']:
            transport = MlxServerClient(settings['model'], base_url=settings['base_url'],
                                        temperature=settings['temperature'],
                                        model_path=settings['model_path'])
        elif settings['model'].startswith('deepseek'):
            transport = SampledDeepSeekClient(settings['model'], temperature=settings['temperature'])
        else:
            transport = make_client(settings['model'], think=True,
                                    options={'temperature': settings['temperature'],
                                             'num_ctx': 32768, 'num_predict': 4096})
        client = ChatUsage(transport, model=settings['model'])
        with SQLiteStorage(INDEX, read_only=True) as storage:
            manifest = storage.active_manifest(VAULT_ID)
            engine = runtime.retrieval_engine(storage, manifest, modes=('bm25', 'semantic'), exact=True)
            tools = runtime.agent_tools(engine=engine, directory=VAULT, vault_id=VAULT_ID, rerank=False,
                                        mode=settings['mode'], prepare_exact=True,
                                        links=LinkGraph(storage, manifest.index_version))
            try:
                for question, sample in tasks:
                    if spent >= settings['max_cost_per_worker']:
                        stopped = 'budget'
                        break
                    observer = AgentObserver(budget=BUDGET, counter=lambda s: len(
                        tokenizer.encode(s, add_special_tokens=False).ids), counter_identity=counter_id)
                    mark, start = len(client.records), perf_counter()
                    result, error = None, None
                    try:
                        with deadline(QUESTION_DEADLINE_SECONDS):
                            result = run_agent(question['question'], tools=tools, model=settings['model'],
                                               max_turns=8, think=True, client=client, observer=observer,
                                               system_instruction=instruction)
                    except Exception as exc:
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                    totals = client.totals(mark)
                    cost = estimate_cost(totals) or 0.0
                    spent += cost
                    row = {'question': question, 'sample': sample, 'elapsed_ms': (perf_counter() - start) * 1000,
                           'error': error, 'result': asdict(result) if result else None,
                           'usage': totals, 'cost_usd': cost}
                    row['scores'] = score(row, question)
                    with rows_path.open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    print(json.dumps({'shard': shard, 'id': question['id'], 'sample': sample,
                                      'status': row['scores']['status'], 'recall': row['scores']['source_recall'],
                                      'tool_calls': row['scores']['tool_calls'], 'cost': round(cost, 5),
                                      'spent': round(spent, 4)}, ensure_ascii=False), flush=True)
            finally:
                tools._exact.close()
                client.client.close()
    (out_dir / f'usage-{shard}.json').write_text(json.dumps(
        {'shard': shard, 'spent_usd': spent, 'stopped': stopped, 'totals': client.totals()}, indent=1) + '\n')


def git_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'training'],
                                    cwd=ROOT, text=True).strip()
    return {'git_head': head, 'dirty': bool(dirty)}


def keep(row, *, min_recall, statuses):
    scores = row['scores']
    recall = scores['source_recall']
    return scores['status'] in statuses and recall is not None and recall >= min_recall


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--questions', type=Path, default=ROOT / 'training/data/questions-pilot.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'training/data/rollouts-pilot')
    parser.add_argument('--model', default='deepseek-reasoner')
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-cost', type=float, default=3.0, help='hard ceiling in USD for the whole run')
    parser.add_argument('--limit', type=int, default=0, help='first N questions only, for a calibration run')
    parser.add_argument('--min-recall', type=float, default=1.0)
    parser.add_argument('--statuses', default='answered,partial')
    parser.add_argument('--base-url', default='', help='serve the model from this OpenAI-compatible endpoint')
    parser.add_argument('--model-path', default=chat_format.MODEL_PATH,
                        help='the served weights, which decide the chat template dialect')
    parser.add_argument('--keep-all', action='store_true',
                        help='do not filter: results.jsonl is every trajectory, for measuring rather than training')
    parser.add_argument('--mode', default=DEFAULT_RETRIEVAL_MODE,
                        help="the agent's default search strategy; follows the product unless given")
    parser.add_argument('--query-hint', default='', help='description to give the search tool\'s query parameter')
    parser.add_argument('--system-instruction', default='', help='a file holding a replacement system instruction')
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text())
    if args.limit:
        questions = questions[:args.limit]
    tasks = [(question, sample) for question in questions for sample in range(args.samples)]
    args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.glob('results-*.jsonl')):
        # Workers append, so a second run under the same directory would double
        # the trajectories and the reported spend without saying so.
        raise SystemExit(f'{args.out} already holds shards; use a new --out.')
    settings = {'model': args.model, 'temperature': args.temperature, 'base_url': args.base_url,
                'mode': args.mode, 'query_hint': args.query_hint, 'model_path': args.model_path,
                'system_instruction': args.system_instruction,
                'max_cost_per_worker': args.max_cost / args.workers}
    (args.out / 'rollout.json').write_text(json.dumps(
        {'questions': len(questions), 'samples': args.samples, 'trajectories': len(tasks), **settings,
         'workers': args.workers, 'max_cost': args.max_cost, 'started_at': strftime('%Y-%m-%dT%H:%M:%S%z')},
        indent=1) + '\n')

    started = perf_counter()
    context = mp.get_context('spawn')
    processes = [context.Process(target=worker, args=(shard, tasks[shard::args.workers], settings, args.out))
                 for shard in range(args.workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [shard for shard, process in enumerate(processes) if process.exitcode]
    if failed:
        print(json.dumps({'workers_failed': failed}), flush=True)

    # Split on newlines, not on str.splitlines(): a vault note can contain
    # U+2028, U+0085 or a vertical tab, which json.dumps writes through
    # unescaped and splitlines() treats as a line break, cutting a record in
    # half. It failed on the sixth trajectory ever generated.
    rows = [json.loads(line) for path in sorted(args.out.glob('results-*.jsonl'))
            for line in path.read_text().split('\n') if line.strip()]
    statuses = tuple(args.statuses.split(','))
    kept = [row for row in rows if keep(row, min_recall=args.min_recall, statuses=statuses)]
    (args.out / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                                    for r in (rows if args.keep_all else kept)))
    (args.out / 'results-all.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    # A run.json beside results.jsonl is what `evaluation.eval compare` reads, so
    # two vault runs compare with the harness's own paired table.
    (args.out / 'run.json').write_text(json.dumps(
        {'label': args.out.name, 'model': args.model, 'think': True, 'questions': len(questions),
         'base_url': args.base_url, 'mode': args.mode, 'query_hint': args.query_hint,
         'system_instruction': args.system_instruction, 'model_path': args.model_path,
         **git_state()}, indent=1) + '\n')
    spent = sum(row.get('cost_usd') or 0 for row in rows)
    report = {'trajectories': len(rows), 'kept': len(kept),
              'retention': round(len(kept) / len(rows), 3) if rows else None,
              'questions_with_a_kept_trajectory': len({r['question']['id'] for r in kept}),
              'questions': len(questions), 'samples': args.samples, 'workers': args.workers,
              'wall_seconds': round(perf_counter() - started, 1), 'spent_usd': round(spent, 4),
              'statuses': dict(Counter(r['scores']['status'] for r in rows)),
              'kept_rule': f'source_recall >= {args.min_recall} and status in {statuses}',
              'workers_failed': failed,
              'model': args.model, 'temperature': args.temperature}
    (args.out / 'rollout-report.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()

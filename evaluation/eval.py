"""Development evaluation: run the product agent on the question set, score it, compare runs.

    python -m evaluation.eval run --label my-change --model qwen3.5:9b --think
    python -m evaluation.eval rescore evaluation/results/my-change
    python -m evaluation.eval compare evaluation/results/baseline evaluation/results/my-change

One question file, one corpus, one script. Expected sources are read only when
scoring, never during a run. Single runs are noisy: read the comparison table
for large, consistent differences, not single wins.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import signal
import subprocess
from threading import current_thread, main_thread
from time import perf_counter, strftime

from arkb.agent.context import ContextPolicy, parse_map_notes
from arkb.agent.observation import AgentBudget, AgentObserver
from arkb.agent.transports import make_client
from arkb.config import RuntimeConfig, load_env_file
from arkb.knowledge.embeddings import tokenizer_fingerprint
from arkb.knowledge.links import LinkGraph
from arkb.knowledge.models import QdrantConfig
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.runtime import Runtime

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / 'evaluation/questions.json'
NOTES = ROOT / 'evaluation/notes'
RESULTS = ROOT / 'evaluation/results'
INDEX = ROOT / '.arkb/eval/index.sqlite'
VAULT_ID = 'eval-notes'
QDRANT_URL = 'http://127.0.0.1:6340'
OPTIONS = {'temperature': 0, 'num_ctx': 32768, 'num_predict': 4096}
BUDGET = AgentBudget(max_tool_calls=12, max_query_calls=10, max_read_calls=6, max_evidence_tokens=8000, max_elapsed_ms=300000)
QUESTION_DEADLINE_SECONDS = 360
TYPES = ('semantic_discovery', 'exploratory_retrieval', 'knowledge_qa', 'multi_hop_qa', 'exact_lookup',
         'direct_read', 'evidence_gap', 'no_retrieval', 'synthesis', 'browse', 'false_premise')
METRICS = ('answered', 'source_recall', 'source_precision', 'delivered_recall', 'complete', 'gap_respected',
           'premise_flagged', 'no_retrieval', 'read_only')
COSTS = ('elapsed_s', 'model_requests', 'tool_calls', 'prompt_tokens', 'eval_tokens', 'evidence_tokens', 'responses_cut')


# --- questions --------------------------------------------------------------

def load_questions(path=QUESTIONS, notes=NOTES):
    questions = json.loads(Path(path).read_text())
    ids = [q['id'] for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate question IDs.')
    sources = {p.name for p in Path(notes).glob('*.md')}
    for q in questions:
        if q['type'] not in TYPES or not q['question'].strip():
            raise ValueError(f"Invalid question {q['id']}.")
        if set(q['expected_sources']) - sources:
            raise ValueError(f"{q['id']} expects a note that is not in {notes}.")
    return questions


# --- transport ------------------------------------------------------------------
# Transport selection and the Ollama client live in arkb.agent.transports, shared with the product.


class DeadlineExceeded(RuntimeError):
    pass


@contextmanager
def deadline(seconds):
    """Hard wall-clock guard around one question, main thread only; the trace still finalizes."""
    if seconds <= 0 or current_thread() is not main_thread():
        raise ValueError('A positive deadline on the main thread is required.')
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise DeadlineExceeded(f'Hard deadline of {seconds:g} seconds exceeded.')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


# --- scoring ----------------------------------------------------------------------

def _hits(event):
    raw = event.get('raw_result') or {}
    hits = [raw['result']] if 'result' in raw else raw.get('results', [])
    return [h for h in hits if isinstance(h, dict)]


def delivered_sources(report):
    """Distinct notes whose evidence entered the conversation."""
    refs = report.get('evidence_references') or {}
    sources = set()
    for event in report.get('tools') or []:
        if not event.get('delivered_to_conversation'):
            continue
        hits = _hits(event)
        allowed = set(event.get('delivered_refs', [h.get('ref') for h in hits]))
        for hit in hits:
            bound = refs.get(hit.get('ref'))
            if bound and hit['ref'] in allowed and (bound.get('content') or '').strip():
                sources.add(bound['source'])
    return sorted(sources)


def score(row, question):
    """Metrics for one saved record; None where a metric does not apply to the question type."""
    result = row.get('result') or {}
    final = result.get('final') or {}
    report = result.get('observation') or {}
    status = final.get('status', 'error')
    tools = [t['name'] for t in report.get('tools') or [] if t.get('executed')]
    cited = sorted({c['source'] for c in final.get('citations') or [] if isinstance(c, dict) and c.get('source')})
    delivered = delivered_sources(report)
    expected = set(question['expected_sources'])
    kind = question['type']
    usage = report.get('usage') or {}
    return {
        'status': status, 'error': row.get('error') or final.get('error'),
        'answer': final.get('answer'), 'cited': cited, 'delivered': delivered, 'tools': tools,
        'answered': status in ('answered', 'partial'),
        'source_recall': len(expected & set(cited)) / len(expected) if expected else None,
        'source_precision': len(expected & set(cited)) / len(cited) if cited else None,
        'delivered_recall': len(expected & set(delivered)) / len(expected) if expected else None,
        'complete': set(cited) == expected if kind == 'exact_lookup' else None,
        # A gap with nothing expected wants an abstention; a gap beside answerable parts wants a partial answer.
        'gap_respected': status == ('partial' if expected else 'insufficient_evidence') if kind == 'evidence_gap' else None,
        # A false premise must not be answered as asked; the honest statuses correct it or report the gap.
        'premise_flagged': status in ('partial', 'insufficient_evidence') if kind == 'false_premise' else None,
        'no_retrieval': not any(t in ('list', 'search', 'match', 'read') for t in tools) if kind == 'no_retrieval' else None,
        'read_only': all(t in ('read', 'finish') for t in tools) if kind == 'direct_read' else None,
        'elapsed_s': row['elapsed_ms'] / 1000, 'model_requests': len(report.get('models') or []), 'tool_calls': len(tools),
        'prompt_tokens': (usage.get('prompt_eval_count') or {}).get('known_total'),
        'eval_tokens': (usage.get('eval_count') or {}).get('known_total'),
        'evidence_tokens': (report.get('evidence') or {}).get('delivered_tokens'),
        'responses_cut': sum(1 for m in report.get('models') or [] if (m.get('response') or {}).get('done_reason') == 'length'),
    }


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    """Overall and per-type means; counts where a rate would hide the denominator."""
    def block(group):
        s = [r['scores'] for r in group]
        out = {'n': len(s), 'errors': sum(x['status'] == 'error' for x in s),
               'statuses': dict(Counter(x['status'] for x in s))}
        for metric in METRICS + COSTS:
            out[metric] = mean([x[metric] for x in s])
        return out
    by_type = defaultdict(list)
    for r in rows:
        by_type[r['question']['type']].append(r)
    return {'all': block(rows), 'by_type': {t: block(g) for t, g in sorted(by_type.items())}}


def _cell(value):
    if value is None:
        return '—'
    return f'{value:,.0f}' if abs(value) >= 100 else f'{value:.3f}' if abs(value) < 10 else f'{value:.1f}'


def markdown(summary, meta):
    lines = [f"# Run `{meta['label']}`", '',
             f"model={meta['model']} think={meta['think']} git={meta['git_head'][:10]}{'+dirty' if meta['dirty'] else ''} "
             f"questions={meta['questions']} wall={meta.get('wall_seconds', 0):.0f}s", '',
             '| type | n | errors | answered | source recall | source precision | delivered recall | complete | gap respected | premise flagged | no retrieval | read only | elapsed s | requests | tool calls | prompt tokens |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, b in [('all', summary['all'])] + list(summary['by_type'].items()):
        cells = [_cell(b[k]) for k in ('answered', 'source_recall', 'source_precision', 'delivered_recall', 'complete', 'gap_respected',
                                       'premise_flagged', 'no_retrieval', 'read_only', 'elapsed_s', 'model_requests', 'tool_calls', 'prompt_tokens')]
        lines.append(f"| {name} | {b['n']} | {b['errors']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


# --- running --------------------------------------------------------------------

def git_state():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'evaluation/eval.py', 'evaluation/questions.json',
                                     'evaluation/notes'], cwd=ROOT, text=True).strip()
    return {'git_head': head, 'dirty': bool(dirty), 'dirty_files': dirty.splitlines()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')


def run(output, *, label, model, think, options=OPTIONS, budget=BUDGET, max_turns=8, limit=0, resume=False,
        effort='high', types=None, questions_path=QUESTIONS, notes=NOTES, policy=ContextPolicy()):
    questions = load_questions(questions_path, notes)
    if types:
        questions = [q for q in questions if q['type'] in types]
    if limit:
        per, kept = Counter(), []
        for q in questions:
            per[q['type']] += 1
            if per[q['type']] <= limit:
                kept.append(q)
        questions = kept
    output.mkdir(parents=True, exist_ok=resume)
    results_path = output / 'results.jsonl'
    rows = []
    if resume and results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        done = {r['question']['id'] for r in rows}
        questions = [q for q in questions if q['id'] not in done]
        meta = json.loads((output / 'run.json').read_text())
        meta.update(resumed_at=strftime('%Y-%m-%dT%H:%M:%S%z'), resumed_from=len(rows))
    else:
        meta = {'label': label, 'model': model, 'think': think, 'effort': effort, 'types': types, 'options': options, 'budget': asdict(budget),
                'context': asdict(policy), 'max_turns': max_turns, 'questions': len(questions),
                'started_at': strftime('%Y-%m-%dT%H:%M:%S%z'), **git_state()}
    write_json(output / 'run.json', meta)
    started = perf_counter()
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(), qdrant_url=QDRANT_URL)) as runtime:
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        # Reused when the notes are unchanged; rebuilt otherwise.
        index = runtime.index(db=INDEX, vault_id=VAULT_ID, notes_dir=notes, qdrant_config=QdrantConfig(url=QDRANT_URL))
        meta['index'] = {'version': index.manifest.index_version, 'documents': index.manifest.document_count, 'reused': index.reused_index}
        tokenizer = runtime.tokenizer()
        counter_id = 'reference-text:' + tokenizer_fingerprint(tokenizer)
        counter = lambda s: len(tokenizer.encode(s, add_special_tokens=False).ids)
        client = make_client(model, options=options, think=think, effort=effort)
        meta['chat_model'] = client.identity
        write_json(output / 'run.json', meta)
        with SQLiteStorage(INDEX, read_only=True) as storage:
            manifest = storage.active_manifest(VAULT_ID)
            engine = runtime.retrieval_engine(storage, manifest, modes=('bm25', 'semantic'), exact=True)
            tools = runtime.agent_tools(engine=engine, directory=notes, vault_id=VAULT_ID, rerank=False,
                                        prepare_exact=True, links=LinkGraph(storage, manifest.index_version))
            try:
                for index_, question in enumerate(questions):
                    observer = AgentObserver(budget=budget, counter=counter, counter_identity=counter_id)
                    start = perf_counter()
                    result, error = None, None
                    try:
                        with deadline(QUESTION_DEADLINE_SECONDS):
                            result = runtime.run_agent(question['question'], tools=tools, model=model, max_turns=max_turns,
                                                       think=think, client=client, observer=observer, policy=policy)
                    except Exception as exc:
                        error = {'type': type(exc).__name__, 'message': str(exc)}
                    row = {'question': question, 'elapsed_ms': (perf_counter() - start) * 1000, 'error': error,
                           'result': asdict(result) if result else None}
                    row['scores'] = score(row, question)
                    rows.append(row)
                    with results_path.open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    print(json.dumps({'done': index_ + 1, 'total': len(questions), 'id': question['id'], 'type': question['type'],
                                      'status': row['scores']['status'], 'elapsed_s': round(row['scores']['elapsed_s'], 1),
                                      'source_recall': row['scores']['source_recall']}), flush=True)
            finally:
                tools._exact.close()
                client.close()
    summary = summarize(rows)
    meta.update(completed_at=strftime('%Y-%m-%dT%H:%M:%S%z'), wall_seconds=meta.get('wall_seconds', 0) + perf_counter() - started)
    write_json(output / 'run.json', meta)
    write_json(output / 'summary.json', summary)
    (output / 'summary.md').write_text(markdown(summary, meta))
    return summary


def rescore(output, questions_path=QUESTIONS, notes=NOTES):
    """Recompute the scores of a saved run with the current scorer and question file."""
    questions = {q['id']: q for q in load_questions(questions_path, notes)}
    rows = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r['question']['id'] in questions]
    for row in rows:
        row['question'] = questions[row['question']['id']]
        row['scores'] = score(row, row['question'])
    (output / 'results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    meta = json.loads((output / 'run.json').read_text())
    meta.update(rescored_at=strftime('%Y-%m-%dT%H:%M:%S%z'))
    summary = summarize(rows)
    write_json(output / 'run.json', meta)
    write_json(output / 'summary.json', summary)
    (output / 'summary.md').write_text(markdown(summary, meta))
    return summary


# --- comparing --------------------------------------------------------------------

def load_run(path):
    path = Path(path)
    rows = {r['question']['id']: r for r in (json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines() if line.strip())}
    return json.loads((path / 'run.json').read_text()), rows


def compare(a, b):
    """Per-question paired differences (B minus A) on shared questions: mean difference and wins/ties/losses."""
    meta_a, rows_a = load_run(a)
    meta_b, rows_b = load_run(b)
    shared = sorted(set(rows_a) & set(rows_b))
    groups = {'all': shared}
    for qid in shared:
        groups.setdefault(rows_a[qid]['question']['type'], []).append(qid)
    table = {}
    for name, ids in groups.items():
        for metric in METRICS + COSTS:
            pairs = [(rows_a[i]['scores'][metric], rows_b[i]['scores'][metric]) for i in ids]
            pairs = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
            if not pairs:
                continue
            va, vb = mean([x for x, _ in pairs]), mean([y for _, y in pairs])
            wins = sum(y > x + 1e-12 for x, y in pairs)
            losses = sum(y < x - 1e-12 for x, y in pairs)
            table[(name, metric)] = {'a': va, 'b': vb, 'diff': vb - va, 'wins': wins, 'ties': len(pairs) - wins - losses,
                                     'losses': losses, 'n': len(pairs)}
    lines = [f"# `{meta_b['label']}` against `{meta_a['label']}`", '',
             f"A: {meta_a['model']} think={meta_a['think']} git={meta_a['git_head'][:10]}{'+dirty' if meta_a['dirty'] else ''}",
             f"B: {meta_b['model']} think={meta_b['think']} git={meta_b['git_head'][:10]}{'+dirty' if meta_b['dirty'] else ''}",
             f'shared questions: {len(shared)}; differences are B minus A; wins/ties/losses count questions where B is better/equal/worse.', '',
             '| group | metric | n | A | B | diff | W/T/L |', '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for (name, metric), v in table.items():
        lines.append(f"| {name} | {metric} | {v['n']} | {_cell(v['a'])} | {_cell(v['b'])} | {v['diff']:+.3f} | {v['wins']}/{v['ties']}/{v['losses']} |")
    return table, '\n'.join(lines) + '\n'


# --- CLI ------------------------------------------------------------------------------

def main():
    load_env_file(ROOT / '.env')  # API keys for the hosted transports; never committed
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run', help='run the product agent on the question set')
    r.add_argument('--label', required=True)
    r.add_argument('--model', default='qwen3.5:9b')
    r.add_argument('--think', action='store_true')
    r.add_argument('--num-ctx', type=int, default=OPTIONS['num_ctx'])
    r.add_argument('--num-predict', type=int, default=OPTIONS['num_predict'])
    r.add_argument('--max-turns', type=int, default=8)
    r.add_argument('--effort', default='high', help='claude-* models with --think: low | medium | high | xhigh | max')
    r.add_argument('--max-evidence-tokens', type=int, default=BUDGET.max_evidence_tokens)
    r.add_argument('--map-note', action='append', default=[], help='candidate layout note; repeatable or comma-separated')
    r.add_argument('--history-tokens', type=int, default=ContextPolicy.history_tokens,
                   help='compact the earliest observations above this estimate; 0 disables it')
    r.add_argument('--limit', type=int, default=0, help='questions per type, for a quick check')
    r.add_argument('--types', default='', help='comma-separated question types to run; default all')
    r.add_argument('--resume', action='store_true', help='continue an interrupted run under the same label')
    r.add_argument('--output', type=Path, help='defaults to evaluation/results/<label>')
    s = sub.add_parser('rescore', help='recompute the scores of a saved run')
    s.add_argument('run', type=Path)
    c = sub.add_parser('compare', help='paired comparison of two runs, B against A')
    c.add_argument('a', type=Path)
    c.add_argument('b', type=Path)
    c.add_argument('--output', type=Path, help='write the Markdown table here as well')
    args = p.parse_args()
    if args.command == 'run':
        options = {**OPTIONS, 'num_ctx': args.num_ctx, 'num_predict': args.num_predict}
        budget = AgentBudget(**{**asdict(BUDGET), 'max_evidence_tokens': args.max_evidence_tokens})
        policy = ContextPolicy(map_notes=parse_map_notes(args.map_note), history_tokens=args.history_tokens)
        summary = run((args.output or RESULTS / args.label).resolve(), label=args.label, model=args.model, think=args.think,
                      options=options, budget=budget, max_turns=args.max_turns, limit=args.limit, resume=args.resume,
                      effort=args.effort, types=[t for t in args.types.split(',') if t] or None, policy=policy)
        print(markdown(summary, json.loads(((args.output or RESULTS / args.label) / 'run.json').read_text())))
    elif args.command == 'rescore':
        summary = rescore(args.run.resolve())
        print(markdown(summary, json.loads((args.run / 'run.json').read_text())))
    else:
        _, text = compare(args.a.resolve(), args.b.resolve())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        print(text)


if __name__ == '__main__':
    main()

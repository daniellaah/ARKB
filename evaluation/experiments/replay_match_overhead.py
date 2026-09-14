"""Replay every recorded pilot match call; no Agent generation or answer scoring."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from arkb.agent.tools import AgentTools, Evidence
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval.engine import RetrievalEngine
from arkb.retrieval.exact import ExactRetriever


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def source_manifest():
    root = Path(__import__('arkb').__file__).parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*.py')}


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = source_manifest()
    write(args.output / 'source.json', sources)
    inputs = json.loads((args.pilot / 'inputs.json').read_text())
    scenarios = json.loads((args.pilot / 'inference/scenarios.json').read_text())
    grouped = defaultdict(list)
    for path in sorted((args.pilot / 'pilot-attempts').glob('*/result.json')):
        record = json.loads(path.read_text())
        complete = json.loads((path.parent / 'complete.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == complete['result_sha256']
        observation = (record.get('result') or {}).get('observation') or {}
        for event in observation.get('tools', []):
            if event['name'] == 'match' and event.get('executed'):
                dataset = record['schedule']['dataset']
                scope = record['scenario_id'] if dataset == 'musique' else dataset
                refs = observation.get('evidence_references') or {}
                expected = [{k: refs[h['ref']].get(k) for k in Evidence.__annotations__}
                            for h in (event.get('raw_result') or {}).get('results', [])]
                grouped[dataset, scope].append((record['key'], record['schedule']['arm'], event, expected))
    rows, preparation, probes = [], [], []
    for (dataset, scope), calls in sorted(grouped.items()):
        entry = scenarios[scope] if dataset == 'musique' else inputs[dataset]
        documents = DocumentAccess(Path(entry['corpus']), vault_id=entry['vault_id'])
        start = perf_counter()
        with ExactRetriever(documents) as exact:
            prepared = exact.prepare()
            preparation.append({'dataset': dataset, 'scope': scope, **prepared,
                                'seconds': perf_counter()-start})
            if dataset != 'musique':
                assert prepared['documents'] == entry['index_manifest']['document_count']
            print(json.dumps({'prepared': dataset, **prepared,
                              'seconds': preparation[-1]['seconds']}), flush=True)
            tools = AgentTools(documents=documents, exact=exact, engine=RetrievalEngine())
            if dataset == 'browsecomp-plus':
                for name, query, options in [
                    ('literal-absent', 'arkb_probe_absent_f5ebd93a78c1', {}),
                    ('literal-common', 'the', {}),
                    ('regex-absent', 'arkb_probe_absent_f5ebd93a78c1', {'regex': True}),
                    ('regex-common', 'the', {'regex': True})]:
                    start = perf_counter()
                    result = exact.search(query, top_k=20, **options)
                    probes.append({'case': name, 'seconds': perf_counter()-start,
                                   'hits': len(result.results), 'result_sha256': digest(asdict(result))})
                write(args.output / 'probes.json', probes)
                print(json.dumps({'probes': probes}), flush=True)
            for key, arm, event, expected in calls:
                start = perf_counter()
                result, error = None, None
                try:
                    result = tools.match(**event['arguments'])
                except Exception as exc:
                    error = {'type': type(exc).__name__, 'message_sha256': digest(str(exc))}
                elapsed = perf_counter()-start
                actual = result['results'] if result else []
                preserved = actual == expected if event['status'] == 'success' and error is None else None
                verified = 0
                for hit in actual:
                    read = documents.read(hit['document_id'], source=hit['source'],
                                          start_char=hit['start_char'], end_char=hit['end_char'])
                    assert (read.content, read.document_revision) == (hit['content'], hit['document_revision'])
                    verified += 1
                row = {'call_id': digest([key, event['index']]), 'dataset': dataset, 'arm': arm,
                       'old_status': event['status'], 'old_error_code': (event.get('error') or {}).get('code'),
                       'old_seconds': (event.get('elapsed_ms') or 0)/1000,
                       'seconds': elapsed, 'error': error, 'hits': len(actual),
                       'previous_success_preserved': preserved, 'live_reference_checks': verified,
                       'result_sha256': digest(result) if result else None}
                rows.append(row)
                write(args.output / 'calls.json', rows)
                if preserved is False:
                    raise AssertionError('Successful historical match result changed: '+row['call_id'])
            write(args.output / 'preparation.json', preparation)
            print(json.dumps({'replayed': dataset, 'calls': len(calls),
                              'errors': sum(r['error'] is not None for r in rows)}), flush=True)
    assert sources == source_manifest(), 'Measured source changed during replay.'
    summary = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'calls': len(rows),
               'errors': sum(r['error'] is not None for r in rows),
               'previous_successes_verified': sum(r['previous_success_preserved'] is True for r in rows),
               'live_reference_checks': sum(r['live_reference_checks'] for r in rows),
               'old_timeout_calls': sum(r['old_error_code'] == 'exact_timeout' for r in rows),
               'timeout_calls_now_completed': sum(r['old_error_code'] == 'exact_timeout' and r['error'] is None for r in rows),
               'source_unchanged': True, 'datasets': {},
               'qualification': 'Development replay of exposed tool calls, not independent Agent quality evaluation. '
                                'Timings include live signature checks and exclude explicit preparation. '
                                'GPU training may share this machine; these are diagnostic timings, not isolated latency results.'}
    for dataset in sorted({r['dataset'] for r in rows}):
        selected = [r for r in rows if r['dataset'] == dataset]
        times = sorted(r['seconds'] for r in selected)
        summary['datasets'][dataset] = {'calls': len(selected), 'mean_seconds': mean(times),
                                       'p95_seconds': times[max(0, __import__('math').ceil(.95*len(times))-1)],
                                       'max_seconds': max(times),
                                       'errors': sum(r['error'] is not None for r in selected)}
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()

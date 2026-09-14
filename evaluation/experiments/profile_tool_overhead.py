"""Read-only full-corpus tool timings and profiles, independent of answer labels."""
import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pstats
from time import perf_counter


def profile(fn):
    profiler = cProfile.Profile()
    start = perf_counter()
    error, result = None, None
    try:
        result = profiler.runcall(fn)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    elapsed = perf_counter() - start
    stats = pstats.Stats(profiler).stats
    rows = [{'file': file, 'line': line, 'function': function, 'calls': values[1],
             'self_seconds': values[2], 'cumulative_seconds': values[3]}
            for (file, line, function), values in stats.items()]
    return {'elapsed_seconds': elapsed, 'error': error,
            'result_sha256': hashlib.sha256(json.dumps(asdict(result), sort_keys=True,
                    ensure_ascii=False).encode()).hexdigest() if result else None,
            'hits': len(result.results) if result else None,
            'profile_by_self': sorted(rows, key=lambda r: -r['self_seconds'])[:25],
            'profile_by_cumulative': sorted(rows, key=lambda r: -r['cumulative_seconds'])[:25]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['match', 'bm25'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-exact', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__import__('arkb').__file__).parent
    manifest = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source.rglob('*.py')}
    (args.output / 'source.json').write_text(json.dumps(manifest, indent=2)+'\n')
    root = Path('/Volumes/ARKBPhaseC')
    start = perf_counter()
    if args.mode == 'match':
        from arkb.knowledge.documents import DocumentAccess
        from arkb.retrieval.exact import ExactRetriever
        tool = ExactRetriever(DocumentAccess(root / 'data/browsecomp-plus/corpus',
                                            vault_id='browsecomp-plus'))
        if args.prepare_exact:
            prepared = tool.prepare()
            (args.output / 'prepared.json').write_text(json.dumps(prepared)+'\n')
        cases = [('literal-absent', 'arkb_probe_absent_f5ebd93a78c1', {}),
                 ('literal-common', 'the', {}),
                 ('regex-absent', 'arkb_probe_absent_f5ebd93a78c1', {'regex': True})]
    else:
        from arkb.knowledge.sqlite import SQLiteStorage
        from arkb.retrieval.bm25 import BM25Retriever
        with SQLiteStorage(root / 'validation/browsecomp-plus/index.sqlite', read_only=True) as storage:
            tool = BM25Retriever.from_snapshot(storage, vault_id='browsecomp-plus')
        cases = [('common-term', 'the', {}),
                 ('multi-term', 'history of science and technology university research', {}),
                 ('rare-term', 'arkb_probe_absent_f5ebd93a78c1', {})]
    setup = perf_counter() - start
    (args.output / 'setup.json').write_text(json.dumps({'seconds': setup})+'\n')
    print(json.dumps({'stage': 'setup_complete', 'seconds': setup}), flush=True)
    for name, query, options in cases:
        result = profile(lambda: tool.search(query, top_k=20, **options))
        (args.output / (name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'case': name, 'seconds': result['elapsed_seconds'],
                          'hits': result['hits'], 'error': result['error']}), flush=True)
    if hasattr(tool, 'close'):
        tool.close()


if __name__ == '__main__':
    main()

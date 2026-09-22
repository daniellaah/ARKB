"""No-LLM engine measurement on the recall scenarios.

For each recall scenario, run the frozen retrieval engine directly in every mode
and report document-level positive recall and distinct sources at 5, 10 and 20
results. This isolates what one search call can deliver from what the agent
chooses to do with it. No labels reach retrieval; scoring happens afterwards.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from arkb.config import RuntimeConfig
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.runtime import Runtime

from .common import ROOT, resolve, write_json
from .run import DEVSET, QDRANT_URL
CUTOFFS = (5, 10, 20)


def measure(devset, output, *, modes=('bm25', 'semantic', 'hybrid')):
    scenarios = [s for s in json.loads((devset / 'scenarios.json').read_text()) if s['slice'].startswith('recall-')]
    labels = json.loads((devset / 'labels.json').read_text())
    scopes = json.loads((devset / 'scopes.json').read_text())
    rows = []
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(),
                               qdrant_url=QDRANT_URL)) as runtime:
        for scope_name in sorted({s['scope'] for s in scenarios}):
            scope = scopes[scope_name]
            source_map = json.loads((devset / 'scoring' / f'{scope_name}.json').read_text())['source_map']
            with SQLiteStorage(resolve(scope['sqlite']), read_only=True) as storage:
                manifest = storage.active_manifest(scope['vault_id'])
                engine = runtime.retrieval_engine(storage, manifest, modes=('bm25', 'semantic'), exact=True)
                for scenario in [s for s in scenarios if s['scope'] == scope_name]:
                    positives = {str(k) for k, v in labels[scenario['id']]['qrels'].items() if v > 0}
                    for mode in modes:
                        start = perf_counter()
                        response = engine.search(scenario['query'], mode=mode, top_k=max(CUTOFFS))
                        seconds = perf_counter() - start
                        ids = [str(source_map[hit.source]) for hit in response.results]
                        row = {'id': scenario['id'], 'dataset': scope_name, 'mode': mode, 'positives': len(positives), 'seconds': seconds}
                        for k in CUTOFFS:
                            seen = list(dict.fromkeys(ids[:k]))
                            row[f'recall@{k}'] = len(set(seen) & positives) / len(positives) if positives else None
                            row[f'distinct_sources@{k}'] = len(seen)
                            row[f'positives_found@{k}'] = len(set(seen) & positives)
                        rows.append(row)
    summary = defaultdict(dict)
    for dataset in sorted({r['dataset'] for r in rows}):
        for mode in modes:
            subset = [r for r in rows if r['dataset'] == dataset and r['mode'] == mode]
            summary[dataset][mode] = {'queries': len(subset), 'seconds_mean': float(np.mean([r['seconds'] for r in subset])),
                                      **{f'recall@{k}': float(np.mean([r[f'recall@{k}'] for r in subset])) for k in CUTOFFS},
                                      **{f'distinct_sources@{k}': float(np.mean([r[f'distinct_sources@{k}'] for r in subset])) for k in CUTOFFS},
                                      **{f'positives_found@{k}': float(np.mean([r[f'positives_found@{k}'] for r in subset])) for k in CUTOFFS}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'engine-recall-rows.json', rows)
    write_json(output / 'engine-recall-summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--devset', type=Path, default=DEVSET)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    summary = measure(args.devset.resolve(), args.output.resolve())
    for dataset, modes in summary.items():
        print(dataset)
        for mode, block in modes.items():
            print(f"  {mode:9s} " + ' '.join(f"r@{k}={block[f'recall@{k}']:.3f} src@{k}={block[f'distinct_sources@{k}']:.1f}" for k in CUTOFFS) + f" t={block['seconds_mean']:.2f}s")


if __name__ == '__main__':
    main()

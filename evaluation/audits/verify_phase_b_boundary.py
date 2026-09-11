"""Expand saved Hybrid provenance and verify the frozen source-pool boundary."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from arkb.evaluation.external import digest, opaque_source, read_jsonl, verify_checksums, write_json
from arkb.knowledge.models import _document_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); verify_checksums(args.run)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'scope': 'Arithmetic audit of already saved P4 leg ranks; no upstream retrieval, embedding or index writes.',
               'pool_manifest_sha256': digest(args.run/'manifest.json'), 'datasets': {}}
    for ds in ('scifact', 'bright-stackoverflow', 'bright-robotics'):
        rows = read_jsonl(args.run/'pools'/f'{ds}.jsonl'); count = 0
        with (args.output/f'{ds}.jsonl').open('x') as stream:
            for row in rows:
                contributions = defaultdict(list); originals = {}
                for leg in sorted(row['hybrid_legs']):
                    seen = set()
                    for rank, hit in enumerate(row['hybrid_legs'][leg], 1):
                        key = (_document_id(ds, hit['source']), 'chunk', hit['chunk_id'])
                        originals.setdefault(key, hit)
                        if key not in seen:
                            contributions[key].append({'leg': leg, 'rank': rank, 'score': hit['score']})
                        seen.add(key)
                scores = {key: math.fsum(1/(60+c['rank']) for c in values)
                          for key, values in contributions.items()}
                order = sorted(scores, key=lambda key: (-scores[key], key))
                selected = []; sources = set()
                for key in order:
                    h = originals[key]
                    if h['source'] in sources: continue
                    sources.add(h['source']); selected.append(key)
                    if len(selected) == 100: break
                mapping = {opaque_source(ds, doc): doc for doc in row['hybrid_ranking']}
                observed = [mapping.get(originals[key]['source']) for key in selected]
                if observed != row['hybrid_ranking']:
                    raise ValueError('Saved Hybrid source order does not reproduce.')
                for i, key in enumerate(selected[:20]):
                    frozen = row['candidates'][i]
                    if frozen['chunk_id'] != key[2] or frozen['score'] != scores[key]:
                        raise ValueError('The rerank pool did not choose the first Hybrid chunk for its source.')
                expanded = [{'original_hybrid_rank': i+1, 'document_id': observed[i],
                             'source_id': key[0], 'source': originals[key]['source'], 'chunk_id': key[2],
                             'start_char': originals[key]['start_char'], 'end_char': originals[key]['end_char'],
                             'hybrid_score': scores[key], 'contributions': contributions[key]}
                            for i, key in enumerate(selected)]
                stream.write(json.dumps({'qid': row['qid'], 'sources': expanded})+'\n')
                count += len(selected)
        summary['datasets'][ds] = {'queries': len(rows), 'source_ranks_verified': count,
                                   'selected_chunk_and_score_checks': len(rows)*20}
        print(ds, summary['datasets'][ds], flush=True)
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'checksums.json', {p.name: digest(p) for p in args.output.iterdir() if p.is_file()})


if __name__ == '__main__':
    main()

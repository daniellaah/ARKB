"""Summarize observed validation retrieval costs from the single saved capture."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from arkb.evaluation.external import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    meta=json.loads((a.run/'experiment.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    if meta['status']!='completed' or digest(a.run/'legs.jsonl.gz')!=meta['legs_sha256']:raise ValueError('Capture incomplete or changed.')
    values={name:[] for name in ('bm25','semantic')};depths={name:[] for name in values};ids=[]
    with gzip.open(a.run/'legs.jsonl.gz','rt') as source:
        for line in source:
            row=json.loads(line);ids.append(row['qid'])
            for leg in values:
                value=row['retrieval_ms'][leg]
                if not np.isfinite(value) or value<0:raise ValueError('Invalid observed retrieval timing.')
                values[leg].append(value);depths[leg].append(len(row['legs'][leg]))
    if ids!=protocol['query_ids']:raise ValueError('Incomplete timing query population.')
    report={'dataset':meta['dataset'],'queries':len(ids),'legs_sha256':meta['legs_sha256'],
        'measurement':'Observed single sequential capture, no timing rerun. Search includes query embedding for Semantic and result hydration; excludes offline fusion and index preparation.',
        'scope':'Full index, 500 chunks/leg, exact retrieval. Not default-depth or Agent latency, and not an isolated repeated latency benchmark.',
        'arms':{name:{'mean_ms':float(np.mean(times)),'p50_ms':float(np.median(times)),
            'p95_ms':float(np.quantile(times,.95)),'mean_returned_chunks':float(np.mean(depths[name]))} for name,times in values.items()}}
    write_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':main()

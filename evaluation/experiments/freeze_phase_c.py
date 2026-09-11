"""Hydrate accepted leg records, replay C0 and diagnose before alternative design."""
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from arkb.evaluation.external import digest, load_external, rank_metrics, aspect_metrics, write_json
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.retrieval.models import chunk_result
from arkb.retrieval.hybrid import rrf

DATASETS = ('scifact', 'bright-stackoverflow', 'bright-robotics')
EXPECTED = (.7216396110820045, .4806525133795389, .3904138067602195)


def emit(f, value):
    f.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')


def unique(hits, limit=None):
    seen=set(); output=[]
    for h in hits:
        if h.source_id not in seen:
            seen.add(h.source_id); output.append(h)
            if len(output)==limit: break
    return output


def stats(values):
    return {'mean':float(np.mean(values)), 'p50':float(np.median(values)),
            'p90':float(np.quantile(values,.9))}


def diagnose(legs, fused, ranking, mapping, qrels):
    pos={d for d,v in qrels.items() if v>0}
    legsets={name:{mapping[h.source] for h in hits} for name,hits in legs.items()}
    union=legsets['bm25']|legsets['semantic']; allhits={h.identity:h for hits in legs.values() for h in hits}
    counts=Counter(mapping[h.source] for h in allhits.values())
    topcounts=Counter(h.source_id for h in fused[:100])
    reps=unique(fused,100); differing=set(); source_conflicts=set()
    for name,hits in legs.items():
        best={h.source_id:h for h in unique(hits)}
        for rep in reps:
            if rep.source_id in best and rep.identity!=best[rep.source_id].identity:
                differing.add(mapping[rep.source]); source_conflicts.add(name+':'+mapping[rep.source])
    groups={
        'positive_absent_both':sorted(pos-union),
        'bm25_only_positive':sorted((pos&legsets['bm25'])-legsets['semantic']),
        'semantic_only_positive':sorted((pos&legsets['semantic'])-legsets['bm25']),
        'shared_positive':sorted(pos&legsets['bm25']&legsets['semantic']),
        'union_positive_lost_top100':sorted((pos&union)-set(ranking[:100])),
        'positive_below_top20':sorted((pos&set(ranking[:100]))-set(ranking[:20])),
        'positive_multiple_chunks':sorted(d for d in pos if counts[d]>1),
        'representative_leg_disagreement':sorted(differing),
        'positive_representative_leg_disagreement':sorted(pos&differing),
    }
    return {'groups':groups, 'union_recall':len(pos&union)/len(pos),
        'leg_recall':{leg:len(pos&ds)/len(pos) for leg,ds in legsets.items()},
        'unique_leg_top100':{leg:len({h.source_id for h in hits[:100]}) for leg,hits in legs.items()},
        'unique_fused_chunks_top100':len(topcounts), 'unique_fused_sources_top100':len(reps),
        'duplicate_competition':any(c>1 for c in topcounts.values()),
        'duplicate_chunk_positions_top100':sum(topcounts.values())-len(topcounts),
        'union_source_count':len(union), 'candidate_chunk_records':sum(map(len,legs.values())),
        'distinct_candidate_chunks':len(allhits), 'chunks_per_returned_source':stats([counts[d] for d in ranking]),
        'representative_leg_disagreements':len(source_conflicts),
        'weak_representative_quality':'unmeasured: source qrels do not label chunk evidence'}


def main():
    root=Path('evaluation/results/phase-c-v1'); freeze=root/'frozen'; freeze.mkdir(exist_ok=False)
    summary={}; manifest={'baseline_commit':'75ba733','starting_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'environment':{'python':sys.version,'platform':platform.platform()},'datasets':{}}
    for ds,expected in zip(DATASETS,EXPECTED):
        data_path=Path('evaluation/results/p4-data')/ds; data=load_external(data_path); mapping=data.source_map()
        out=freeze/ds; out.mkdir(); write_json(out/'scoring.json',{'qrels':data.qrels,'aspects':data.aspects})
        write_json(out/'queries.json',data.queries); write_json(out/'source-map.json',mapping)
        upstream=Path('evaluation/results/phase-b-v1/pools')/(ds+'.jsonl')
        indexed=Path('evaluation/results')/('p4-'+ds+'-indexed'); perquery=[]
        with SQLiteStorage(indexed/'index.sqlite',read_only=True) as storage:
            snapshot=storage.active_manifest(ds); records={r.chunk_id:r for r in storage.snapshot_records(snapshot.index_version)}
            config={'snapshot':asdict(snapshot),'build':storage.build_metadata(snapshot.index_version),
                'rrf_k':60,'leg_depth':500,'top_sources':100,'rerank':False,'upstream_pool_sha256':digest(upstream),
                'corpus_sha256':digest(data_path/'corpus.jsonl'),'data_manifest':json.loads((data_path/'manifest.json').read_text()),
                'upstream_protocol':json.loads((indexed/'protocol.json').read_text()),
                'baseline_source_sha256':digest('src/arkb/retrieval/hybrid.py')}
            write_json(out/'config.json',config)
            with upstream.open() as source, gzip.open(out/'bm25.jsonl.gz','wt',compresslevel=1) as b, gzip.open(out/'semantic.jsonl.gz','wt',compresslevel=1) as s, (out/'c0.jsonl').open('x') as c:
                for i,line in enumerate(source):
                    row=json.loads(line); assert row['query']==data.queries[row['qid']]
                    legs={}
                    for name,f in (('bm25',b),('semantic',s)):
                        hits=[]
                        for rank,raw in enumerate(row['hybrid_legs'][name],1):
                            h=chunk_result(records[raw['chunk_id']],method=name,index_id=snapshot.index_version,
                                score=raw['score'],score_type='bm25' if name=='bm25' else 'cosine_similarity')
                            assert (h.source,h.start_char,h.end_char)==(raw['source'],raw['start_char'],raw['end_char'])
                            hits.append(h)
                        legs[name]=hits
                        emit(f,{'dataset':ds,'qid':row['qid'],'query':row['query'],'index_id':snapshot.index_version,
                            'results':[{'rank':j+1,'hit':asdict(h)} for j,h in enumerate(hits)]})
                    fused=rrf(legs,k=60); selected=unique(fused,100); ranking=[mapping[h.source] for h in selected]
                    assert ranking==row['hybrid_ranking'],(ds,row['qid'],'rank mismatch')
                    for h,old in zip(selected[:20],row['candidates']):
                        assert (h.chunk_id,h.score,h.content,h.metadata['document_revision'])==(old['chunk_id'],old['score'],old['content'],old['metadata']['document_revision'])
                    metrics={**rank_metrics(data.qrels[row['qid']],ranking,ks=(10,20,100)),**aspect_metrics(data.aspects.get(row['qid'],[]),ranking)}
                    diag=diagnose(legs,fused,ranking,mapping,data.qrels[row['qid']])
                    result={'qid':row['qid'],'ranking':ranking,'representatives':[{'source_id':h.source_id,'chunk_id':h.chunk_id,'score':h.score} for h in selected],
                        'metrics':metrics,'diagnostics':diag}
                    perquery.append(result);emit(c,result)
                    if i%50==0: print(ds,i+1,flush=True)
        metrics={m:float(np.mean([r['metrics'][m] for r in perquery])) for m in perquery[0]['metrics'] if perquery[0]['metrics'][m] is not None}
        assert abs(metrics['ndcg@10']-expected)<1e-14
        diags=[r['diagnostics'] for r in perquery]
        summary[ds]={'queries':len(perquery),'exact_source_ranks':sum(len(r['ranking']) for r in perquery),'exact_representative_checks':len(perquery)*20,'metrics':metrics,
            'group_query_counts':{g:sum(bool(d['groups'][g]) for d in diags) for g in diags[0]['groups']},
            'group_positive_counts':{g:sum(len(d['groups'][g]) for d in diags) for g in diags[0]['groups']},
            'union_recall':float(np.mean([d['union_recall'] for d in diags])),
            'leg_recall':{leg:float(np.mean([d['leg_recall'][leg] for d in diags])) for leg in ('bm25','semantic')},
            'duplicate_competition_queries':sum(d['duplicate_competition'] for d in diags),
            'unique_leg_top100':{leg:stats([d['unique_leg_top100'][leg] for d in diags]) for leg in ('bm25','semantic')},
            'unique_fused_chunks_top100':stats([d['unique_fused_chunks_top100'] for d in diags]),
            'chunks_per_returned_source':{m:float(np.mean([d['chunks_per_returned_source'][m] for d in diags])) for m in ('mean','p50','p90')}}
        write_json(out/'checksums.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})
        manifest['datasets'][ds]={'checksums_sha256':digest(out/'checksums.json'),'query_ids':list(data.queries)}
        print(ds,summary[ds],flush=True)
    write_json(freeze/'manifest.json',manifest);write_json(root/'c0-diagnostics.json',summary)

if __name__=='__main__':main()

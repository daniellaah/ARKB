"""Replay preregistered fusion policies over immutable legs; no model calls."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
from arkb.evaluation.external import digest,verify_checksums,write_json,rank_metrics,aspect_metrics,reference_metrics
from arkb.evaluation.source_fusion import fuse_sources
from arkb.retrieval.models import SearchResult,SearchResponse
from arkb.retrieval.hybrid import rrf
from freeze_phase_c import DATASETS,unique,emit,stats


def load_legs(directory):
    with gzip.open(directory/'bm25.jsonl.gz','rt') as b,gzip.open(directory/'semantic.jsonl.gz','rt') as s:
        for left,right in zip(b,s,strict=True):
            rows={'bm25':json.loads(left),'semantic':json.loads(right)}
            assert rows['bm25']['qid']==rows['semantic']['qid']
            yield rows['bm25']['qid'],{name:SearchResponse(query=r['query'],method=name,index_id=r['index_id'],
                results=tuple(SearchResult(**h['hit']) for h in r['results'])) for name,r in rows.items()}


def run_policy(legs,policy,top_k=100):
    if policy=='C0':return tuple(unique(rrf({name:r.results for name,r in legs.items()},k=60),top_k))
    if policy in ('bm25','semantic'):return tuple(unique(legs[policy].results,top_k))
    return fuse_sources(legs,policy=policy,top_k=top_k).results


def paired(left,right):
    delta=np.array(right)-np.array(left); rng=np.random.default_rng(20260911)
    boot=np.mean(rng.choice(delta,size=(10000,len(delta)),replace=True),axis=1)
    return {'mean_delta':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
        'wins':int((delta>0).sum()),'ties':int((delta==0).sum()),'losses':int((delta<0).sum())}


def compare_diagnostics(base,other,positives):
    a={d:i+1 for i,d in enumerate(base)};b={d:i+1 for i,d in enumerate(other)}
    common=set(a)&set(b)
    return {'recovered_positive':sorted((set(b)-set(a))&positives),
        'lost_positive':sorted((set(a)-set(b))&positives),
        'positive_promotions':sorted(d for d in positives if b.get(d,101)<a.get(d,101)),
        'positive_demotions':sorted(d for d in positives if b.get(d,101)>a.get(d,101)),
        'mean_common_source_rank_movement':float(np.mean([a[d]-b[d] for d in common])) if common else 0.,
        'new_nonpositive_top10':sorted((set(other[:10])-set(base[:10]))-positives),
        'recovered_positive_top10':sorted((set(other[:10])-set(base[:10]))&positives),
        'lost_positive_top10':sorted((set(base[:10])-set(other[:10]))&positives)}


def redundancy_gate(c1,c2,legs,mapping,positives):
    lost=(set(c1[:10])-set(c2[:10]))&positives
    if not lost:return []
    promoted=(set(c2[:10])-set(c1[:10]))-positives
    repeated=[]
    for name,response in legs.items():
        sources={}
        for h in response.results:
            ds=mapping[h.source]
            if ds in promoted:
                saved=sources.setdefault(ds,[])
                if h.identity not in {x.identity for x in saved} and len(saved)<2:saved.append(h)
        for ds,hits in sources.items():
            if len(hits)==2:
                a,b=[set(h.content.split()) for h in hits]
                similarity=len(a&b)/len(a|b) if a|b else 0
                if similarity>=.9:repeated.append({'document_id':ds,'leg':name,'jaccard':similarity,
                    'chunks':[h.chunk_id for h in hits],'lost_positive_top10':sorted(lost)})
    return repeated


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path('evaluation/results/phase-c-v1/frozen');summaries={}
    protocol=Path('evaluation/phase-c/v1/protocol.json')
    write_json(args.output/'run-config.json',{'started_at':datetime.now(timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':digest('src/arkb/evaluation/source_fusion.py'),'runner_sha256':digest(__file__),
        'frozen_manifest_sha256':digest(root/'manifest.json'),'protocol_sha256':digest(protocol),'rerank':False})
    for ds in DATASETS:
        directory=root/ds;verify_checksums(directory)
        scoring=json.loads((directory/'scoring.json').read_text());mapping=json.loads((directory/'source-map.json').read_text())
        baseline={r['qid']:r for r in map(json.loads,(directory/'c0.jsonl').read_text().splitlines())}
        rows=[];arms=['C0','C1','C2','bm25','semantic']
        with gzip.open(args.output/(ds+'.jsonl.gz'),'wt',compresslevel=1) as stream:
            for i,(qid,legs) in enumerate(load_legs(directory)):
                outputs={};times={arm:[] for arm in arms[:3]}
                if i==0:
                    for arm in arms[:3]:run_policy(legs,arm)
                for rep in range(3):
                    order=arms[:3][(i+rep)%3:]+arms[:3][:(i+rep)%3]
                    for arm in order:
                        start=perf_counter();outputs[arm]=run_policy(legs,arm);times[arm].append((perf_counter()-start)*1000)
                for arm in arms[3:]:outputs[arm]=run_policy(legs,arm)
                rankings={arm:[mapping[h.source] for h in hits] for arm,hits in outputs.items()}
                assert rankings['C0']==baseline[qid]['ranking']
                positives={d for d,v in scoring['qrels'][qid].items() if v>0}
                gate=redundancy_gate(rankings['C1'],rankings['C2'],legs,mapping,positives)
                row={'qid':qid,'arms':{},'c3_redundant_promotion_evidence':gate,
                    'candidate_chunks_processed':sum(len(r.results) for r in legs.values()),
                    'union_unique_sources':len({h.source_id for r in legs.values() for h in r.results})}
                for arm,hits in outputs.items():
                    metrics={**rank_metrics(scoring['qrels'][qid],rankings[arm],ks=(10,20,100)),
                             **aspect_metrics(scoring['aspects'].get(qid,[]),rankings[arm])}
                    row['arms'][arm]={'ranking':rankings[arm],'metrics':metrics,'fusion_ms':times.get(arm),
                        'diagnostics':compare_diagnostics(rankings['C0'],rankings[arm],positives),
                        'representative_changed_common_sources':sum(
                            h.chunk_id!=next(x.chunk_id for x in outputs['C0'] if x.source_id==h.source_id)
                            for h in hits if h.source_id in {x.source_id for x in outputs['C0']}),
                        'representatives':[{'source_id':h.source_id,'source':h.source,'chunk_id':h.chunk_id,
                            'start_char':h.start_char,'end_char':h.end_char,'score':h.score,'metadata':h.metadata} for h in hits]}
                rows.append(row);emit(stream,row)
                if i%50==0:print(ds,i+1,flush=True)
        metrics=[m for m,v in rows[0]['arms']['C0']['metrics'].items() if v is not None]
        summary={'queries':len(rows),'arms':{},'paired_vs_C0':{},'c3_gate':{},'source_leg_checksums_sha256':digest(directory/'checksums.json')}
        for arm in arms:
            summary['arms'][arm]={'metrics':{m:float(np.mean([r['arms'][arm]['metrics'][m] for r in rows])) for m in metrics},
                'diagnostic_query_counts':{key:sum(bool(r['arms'][arm]['diagnostics'][key]) for r in rows)
                    for key in rows[0]['arms'][arm]['diagnostics'] if key!='mean_common_source_rank_movement'},
                'diagnostic_source_counts':{key:sum(len(r['arms'][arm]['diagnostics'][key]) for r in rows)
                    for key in rows[0]['arms'][arm]['diagnostics'] if key!='mean_common_source_rank_movement'},
                'mean_common_source_rank_movement':float(np.mean([r['arms'][arm]['diagnostics']['mean_common_source_rank_movement'] for r in rows])),
                'mean_representative_changes':float(np.mean([r['arms'][arm]['representative_changed_common_sources'] for r in rows])),
                'mean_sources_returned':float(np.mean([len(r['arms'][arm]['ranking']) for r in rows]))}
            if arm in arms[:3]:
                latencies=[np.mean(r['arms'][arm]['fusion_ms']) for r in rows]
                summary['arms'][arm]['fusion_ms']={'mean':float(np.mean(latencies)),'p50':float(np.median(latencies)),
                    'p95':float(np.quantile(latencies,.95))}
                summary['arms'][arm]['mean_candidate_chunks']=float(np.mean([r['candidate_chunks_processed'] for r in rows]))
            if arm in ('C1','C2'):
                summary['paired_vs_C0'][arm]={m:paired([r['arms']['C0']['metrics'][m] for r in rows],
                    [r['arms'][arm]['metrics'][m] for r in rows]) for m in metrics}
            reference=reference_metrics(scoring['qrels'],{r['qid']:r['arms'][arm]['ranking'] for r in rows})
            maxerror=max(abs(value-r['arms'][arm]['metrics'][m]) for r in rows for m,value in reference[r['qid']].items())
            assert maxerror<1e-9;summary['arms'][arm]['reference_max_error']=maxerror
        diff=summary['arms']['C2']['metrics']['ndcg@10']-summary['arms']['C1']['metrics']['ndcg@10']
        repeats=sum(bool(r['c3_redundant_promotion_evidence']) for r in rows)
        summary['c3_gate']={'c2_minus_c1_ndcg10':diff,'redundant_promotion_queries':repeats,'query_fraction':repeats/len(rows),
            'passes':ds.startswith('bright-') and diff<=-.01 and repeats/len(rows)>=.1}
        summaries[ds]=summary;write_json(args.output/'summary.json',summaries)
        print(ds,{arm:v['metrics']['ndcg@10'] for arm,v in summary['arms'].items()},summary['c3_gate'],flush=True)
    write_json(args.output/'checksums.json',{p.name:digest(p) for p in args.output.iterdir() if p.is_file()})

if __name__=='__main__':main()

"""Replay saved validation legs, source ranks, evidence and all official metrics offline."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from freeze_phase_c import unique
from arkb.evaluation.external import digest,load_external,verify_checksums,write_json,rank_metrics,reference_metrics
from arkb.retrieval.hybrid import rrf
from arkb.retrieval.models import SearchResult


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    verified=verify_checksums(a.run);verify_checksums(a.dataset);data=load_external(a.dataset);mapping=data.source_map()
    meta=json.loads((a.run/'experiment.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    summary=json.loads((a.run/'summary.json').read_text());decision=json.loads((a.run/'development-decision.json').read_text())
    assert meta['status']=='completed' and decision['selected_policy']=='C0'
    assert meta['dataset_manifest_sha256']==protocol['dataset_manifest_sha256']==digest(a.dataset/'manifest.json')
    if 'input_checksums_sha256' in protocol:
        assert protocol['input_checksums_sha256']==digest(a.run/'input-checksums.json')==digest(a.dataset/'checksums.json')
    assert meta['snapshot_before']==meta['snapshot_after']
    assert meta['legs_sha256']==digest(a.run/'legs.jsonl.gz')
    assert meta['index_sqlite_sha256']==digest(a.run/'index.sqlite')
    labels={'evidence':data.qrels} if data.name=='browsecomp-plus' else {'qrels':data.qrels}
    if (a.dataset/'gold-qrels.json').exists():labels['gold']=json.loads((a.dataset/'gold-qrels.json').read_text())
    ks=(5,10,20,100,1000) if 'gold' in labels else (10,20,100)
    leg_hashes={name:hashlib.sha256() for name in ('bm25','semantic')}
    rankings={name:{} for name in ('bm25','semantic','C0')};metrics={name:[] for name in rankings}
    query_ids=[];rank_checks=evidence_checks=0
    with gzip.open(a.run/'legs.jsonl.gz','rt') as legs_file,gzip.open(a.run/'rows.jsonl.gz','rt') as rows_file:
        for line in legs_file:
            raw=json.loads(line);qid=raw['qid'];query_ids.append(qid)
            assert raw['query']==data.queries[qid] and raw['index_id']==meta['index_report']['manifest']['index_version']
            legs={name:[SearchResult(**r['hit']) for r in hits] for name,hits in raw['legs'].items()}
            for name,hits in raw['legs'].items():
                assert [r['rank'] for r in hits]==list(range(1,len(hits)+1)) and len(hits)<=500
                leg_hashes[name].update((json.dumps({'qid':qid,'query':raw['query'],'results':hits},sort_keys=True,ensure_ascii=False)+'\n').encode())
                for hit in legs[name]:
                    assert hit.metadata['index_version']==raw['index_id']
                    assert hit.source in mapping and hit.end_char-hit.start_char==len(hit.content)
            arms={**{n:unique(h,1000) for n,h in legs.items()},'C0':unique(rrf(legs,k=60),1000)}
            for arm,hits in arms.items():
                row=json.loads(next(rows_file));ranking=[mapping[h.source] for h in hits]
                assert (row['qid'],row['arm'],row['ranking'])==(qid,arm,ranking)
                assert len(set(ranking))==len(ranking)==row['returned_sources']
                assert row['candidate_chunks_processed']==sum(map(len,legs.values()))
                value={label:rank_metrics(qrels[qid],ranking,ks=ks) for label,qrels in labels.items()}
                assert value==row['metrics'];metrics[arm].append(value);rankings[arm][qid]=ranking
                rank_checks+=len(ranking)
                for hit,saved in zip(hits,row['representatives'],strict=True):
                    assert saved=={'document_id':mapping[hit.source],'source_id':hit.source_id,'chunk_id':hit.chunk_id,
                        'start_char':hit.start_char,'end_char':hit.end_char,'document_revision':hit.metadata['document_revision'],
                        'score':hit.score}
                    evidence_checks+=1
        assert next(rows_file,None) is None
    assert query_ids==protocol['query_ids']==list(data.queries)
    assert {n:h.hexdigest() for n,h in leg_hashes.items()}==meta['leg_sha256']
    max_error=0.
    for arm,values in metrics.items():
        expected_lines=[f'{qid} Q0 {doc} {i} {len(ranking)-i+1} ARKB-{arm}\n'
            for qid,ranking in rankings[arm].items() for i,doc in enumerate(ranking,1)]
        assert (a.run/(arm+'.trec')).read_text()==''.join(expected_lines)
        for label,qrels in labels.items():
            ref=reference_metrics(qrels,rankings[arm],ks=ks)
            for qid,value in zip(query_ids,values,strict=True):
                max_error=max(max_error,max(abs(v-value[label][m]) for m,v in ref[qid].items()))
            for metric,expected in summary['arms'][arm]['labels'][label]['metrics'].items():
                actual=sum(v[label][metric] for v in values)/len(values)
                assert abs(actual-expected)<1e-14
                assert summary['paired_selected_minus_C0'][label][metric]=={
                    'mean_delta':0.,'ci95':[0.,0.],'wins':0,'ties':len(values),'losses':0}
    assert max_error<1e-9
    write_json(a.output,{'status':'passed','dataset':data.name,'queries':len(query_ids),
        'verified_run_files':verified,'exact_source_rank_positions':rank_checks,
        'exact_representative_checks':evidence_checks,'reference_metric_max_error':max_error,
        'leg_sha256':meta['leg_sha256'],'model_calls':0,'upstream_retrieval_calls':0,
        'run_checksums_sha256':digest(a.run/'checksums.json'),'script_sha256':digest(__file__),
        'selected_alias':'C0: identity comparison, not an independent treatment run'})
    print(data.name,'offline validation replay passed',len(query_ids),rank_checks)


if __name__=='__main__':main()

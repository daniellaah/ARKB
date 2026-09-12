"""Offline exact replay of Phase C development ranks, evidence and paired statistics."""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_phase_c import load_legs,run_policy,paired
from arkb.evaluation.external import digest,verify_checksums,write_json,rank_metrics,aspect_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    verify_checksums(a.development);summary=json.loads((a.development/'summary.json').read_text());report={}
    for ds,expected in summary.items():
        root=a.frozen/ds;verified=verify_checksums(root)
        mapping=json.loads((root/'source-map.json').read_text());queries=json.loads((root/'queries.json').read_text())
        labels=json.loads((root/'scoring.json').read_text());config=json.loads((root/'config.json').read_text())
        metrics_by_arm={arm:[] for arm in ('C0','C1','C2','bm25','semantic')};rank_checks=evidence_checks=0;ids=[]
        with gzip.open(a.development/(ds+'.jsonl.gz'),'rt') as saved:
            for (qid,legs),line in zip(load_legs(root),saved,strict=True):
                row=json.loads(line);assert row['qid']==qid;ids.append(qid)
                assert all(r.query==queries[qid] and r.index_id==config['snapshot']['index_version'] for r in legs.values())
                for arm in metrics_by_arm:
                    hits=run_policy(legs,arm);ranking=[mapping[h.source] for h in hits];record=row['arms'][arm]
                    assert ranking==record['ranking'],(ds,qid,arm,'rank')
                    rank_checks+=len(ranking)
                    metrics={**rank_metrics(labels['qrels'][qid],ranking,ks=(10,20,100)),**aspect_metrics(labels['aspects'].get(qid,[]),ranking)}
                    assert metrics==record['metrics'],(ds,qid,arm,'metric')
                    metrics_by_arm[arm].append(metrics)
                    for hit,rep in zip(hits,record['representatives'],strict=True):
                        assert (hit.source_id,hit.source,hit.chunk_id,hit.start_char,hit.end_char,hit.score,hit.metadata)==(
                            rep['source_id'],rep['source'],rep['chunk_id'],rep['start_char'],rep['end_char'],rep['score'],rep['metadata'])
                        evidence_checks+=1
        assert ids==list(queries)
        stats_checks=0
        for arm in ('C1','C2'):
            for name,expected_stat in expected['paired_vs_C0'][arm].items():
                actual=paired([m[name] for m in metrics_by_arm['C0']],[m[name] for m in metrics_by_arm[arm]])
                assert actual==expected_stat,(ds,arm,name,'paired')
                stats_checks+=1
        for arm,values in metrics_by_arm.items():
            for name,value in expected['arms'][arm]['metrics'].items():
                actual=sum(m[name] for m in values)/len(values)
                assert abs(actual-value)<1e-14,(ds,arm,name,'aggregate')
        report[ds]={'queries':len(ids),'verified_input_files':verified,'exact_rank_positions':rank_checks,
            'exact_representative_and_provenance_checks':evidence_checks,'paired_statistic_replays':stats_checks}
    write_json(a.output,{'status':'passed','datasets':report,'model_calls':0,'upstream_retrieval_calls':0,
        'frozen_manifest_sha256':digest(a.frozen/'manifest.json'),'development_checksums_sha256':digest(a.development/'checksums.json')})
    print(json.dumps(report))

if __name__=='__main__':main()

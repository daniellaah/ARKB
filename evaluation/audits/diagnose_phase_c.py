"""Additional source/representative diagnostics from saved development results."""
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from run_phase_c import load_legs
from arkb.evaluation.external import digest,verify_checksums,write_json


def main():
    root=Path('evaluation/results/phase-c-v1');out={}
    for ds in ('scifact','bright-stackoverflow','bright-robotics'):
        frozen=root/'frozen'/ds;verify_checksums(frozen)
        mapping=json.loads((frozen/'source-map.json').read_text());scoring=json.loads((frozen/'scoring.json').read_text())
        rows=[]
        with gzip.open(root/'development'/(ds+'.jsonl.gz'),'rt') as results:
            for (qid,legs),line in zip(load_legs(frozen),results,strict=True):
                row=json.loads(line);assert qid==row['qid']
                hits={h.identity:h for response in legs.values() for h in response.results}
                counts=Counter(mapping[h.source] for h in hits.values());union=set(counts)
                positives={d for d,v in scoring['qrels'][qid].items() if v>0}
                arms={}
                for arm in ('C0','C1','C2'):
                    ranking=row['arms'][arm]['ranking'];reps=row['arms'][arm]['representatives']
                    conflicts=[];valid=0
                    for rep in reps:
                        h=hits[(rep['source_id'],'chunk',rep['chunk_id'])]
                        assert (rep['source'],rep['start_char'],rep['end_char'],rep['metadata']['document_revision'],rep['metadata']['index_version'])==(
                            h.source,h.start_char,h.end_char,h.metadata['document_revision'],h.metadata['index_version'])
                        valid+=1
                        for name,response in legs.items():
                            best=next((x for x in response.results if x.source_id==h.source_id),None)
                            if best is not None and best.identity!=h.identity:conflicts.append({'document_id':mapping[h.source],'leg':name})
                    top_scores=Counter(r['score'] for r in reps[:10])
                    arms[arm]={'unique_sources_top100':len(set(ranking)),'chunks_per_returned_source':[counts[d] for d in ranking],
                        'union_positive_lost_top100':sorted((positives&union)-set(ranking)),
                        'positive_below_top20':sorted((positives&set(ranking))-set(ranking[:20])),
                        'representative_leg_disagreements':conflicts,'representative_span_revision_checks':valid,
                        'top10_exact_score_tie':any(c>1 for c in top_scores.values()),
                        'chunk_quality':'not identifiable from source qrels'}
                rows.append({'qid':qid,'arms':arms})
        arms={}
        for arm in ('C0','C1','C2'):
            records=[r['arms'][arm] for r in rows];counts=[c for r in records for c in r['chunks_per_returned_source']]
            arms[arm]={'mean_unique_sources_top100':float(np.mean([r['unique_sources_top100'] for r in records])),
                'chunks_per_returned_source_pooled_within_dataset':{'mean':float(np.mean(counts)),'p50':float(np.median(counts)),'p90':float(np.quantile(counts,.9))},
                'union_positive_lost_queries':sum(bool(r['union_positive_lost_top100']) for r in records),
                'union_positive_lost_sources':sum(len(r['union_positive_lost_top100']) for r in records),
                'positive_below_top20_queries':sum(bool(r['positive_below_top20']) for r in records),
                'representative_disagreement_queries':sum(bool(r['representative_leg_disagreements']) for r in records),
                'representative_leg_disagreements':sum(len(r['representative_leg_disagreements']) for r in records),
                'representative_span_revision_checks':sum(r['representative_span_revision_checks'] for r in records),
                'top10_exact_score_tie_queries':sum(r['top10_exact_score_tie'] for r in records)}
        out[ds]=arms
        with gzip.open(root/(ds+'-extra-diagnostics.jsonl.gz'),'wt',compresslevel=1) as f:
            for r in rows:f.write(json.dumps(r)+'\n')
    write_json(root/'extra-diagnostics.json',out);write_json(Path('evaluation/phase-c/v1/extra-diagnostics.json'),out)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()

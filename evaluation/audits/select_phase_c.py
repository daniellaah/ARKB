"""Apply the preregistered shared-policy gate before broader validation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from arkb.evaluation.external import digest,write_json,verify_checksums


def select(summary):
    if any(s['c3_gate']['passes'] for s in summary.values()):
        raise ValueError('C3 gate passed; the preregistered conditional experiment must finish first.')
    checks={}
    for policy in ('C1','C2'):
        paired={ds:r['paired_vs_C0'][policy] for ds,r in summary.items()}
        bright=[ds for ds in paired if ds.startswith('bright-')]
        checks[policy]={
            'scifact_ndcg_preserved':paired['scifact']['ndcg@10']['mean_delta']>=-.01,
            'bright_ndcg_preserved':all(paired[ds]['ndcg@10']['mean_delta']>=-.005 for ds in bright),
            'recall_preserved':all(p[m]['mean_delta']>=-.01 for p in paired.values() for m in ('recall@10','recall@20','recall@100')),
            'bright_aspects_preserved':all(paired[ds][m]['mean_delta']>=-.01 for ds in bright for m in ('alpha_ndcg@10','aspect_recall@10')),
            'convincing_bright_gain':any(paired[ds]['ndcg@10']['mean_delta']>=.01 and paired[ds]['ndcg@10']['ci95'][0]>0 for ds in bright)}
    selected=next((p for p,c in checks.items() if all(c.values())),'C0')
    return {'selected_policy':selected,
        'checks':checks,'c3':'skipped: no domain met the preregistered redundant-chunk domination gate',
        'production_decision':'A: keep current chunk-level Hybrid fusion' if selected=='C0' else 'pending broader validation',
        'validation_rule':'Evaluate once; no further architecture tuning; selected C0 shares the control run.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('development',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify_checksums(a.development)
    result=select(json.loads((a.development/'summary.json').read_text()))
    result.update(frozen_at=datetime.now(timezone.utc).isoformat(),development_summary_sha256=digest(a.development/'summary.json'),
        protocol_sha256=digest('evaluation/phase-c/v1/protocol.json'),fusion_source_sha256=digest('src/arkb/evaluation/source_fusion.py'))
    if a.output.exists():raise ValueError('Development decision is already frozen.')
    write_json(a.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()

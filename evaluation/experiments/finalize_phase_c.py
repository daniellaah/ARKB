"""Verify and archive completed registered validation jobs; no retrieval or tuning."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from arkb.evaluation.external import digest,verify_checksums,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--storage',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];v=root/'evaluation/phase-c/v1';status=a.storage/'validation/finalization-status.json'
    meta={'status':'waiting_for_validation','started_at':datetime.now(timezone.utc).isoformat()};write_json(status,meta)
    try:
        while True:
            completion=json.loads((a.storage/'validation/completion-status.json').read_text())
            if completion['status']=='failed':raise RuntimeError('Validation workflow failed; inspect its preserved status/log.')
            if completion['status']=='completed':break
            time.sleep(30)
        meta['status']='preserving_indexes';write_json(status,meta)
        runs={'nfcorpus':'nfcorpus-r2','fiqa':'fiqa-r2','browsecomp-plus':'browsecomp-plus'}
        locations=json.loads((v/'artifact-locations.json').read_text())
        for ds,run_name in runs.items():
            run=a.storage/'validation'/run_name;data=a.storage/'data'/ds
            experiment=json.loads((run/'experiment.json').read_text())
            replay=json.loads((v/'validation'/(ds+'-replay.json')).read_text())
            if experiment['status']!='completed' or replay['status']!='passed':raise ValueError('Incomplete required evidence: '+ds)
            verify_checksums(run);verify_checksums(data)
            preserved=v/'validation'/(ds+'-preserved-index.json')
            if not preserved.exists():
                subprocess.run([sys.executable,str(root/'evaluation/experiments/preserve_phase_c_snapshot.py'),
                    '--run',str(run),'--dataset',str(data),'--output',str(a.storage/'snapshots'/ds),
                    '--manifest',str(preserved)],check=True,cwd=root)
            index=json.loads(preserved.read_text())
            if digest(index['qdrant_snapshot'])!=index['snapshot_sha256']:raise ValueError('External snapshot changed: '+ds)
            if digest(run/'index.sqlite')!=index['sqlite_sha256']:raise ValueError('SQLite snapshot changed: '+ds)
            locations[ds]={'run':str(run),'checksums_sha256':digest(run/'checksums.json'),
                'snapshot_manifest':'validation/'+preserved.name,'replay':'validation/'+ds+'-replay.json'}
        verification=json.loads((v/'verification/test-summary.json').read_text());verify_checksums(v/'verification')
        for group in verification.values():
            if isinstance(group,dict) and 'cases' in group:
                if group['passed']!=group['cases']:raise ValueError('Regression suite did not completely pass.')
        if verification['freshness']['passed']!=16 or verification['freshness']['failed']:raise ValueError('Freshness not verified.')
        audit=json.loads((v/'scope-audit.json').read_text())
        for name in subprocess.check_output(['git','ls-tree','-r','--name-only','75ba733','src'],text=True,cwd=root).splitlines():
            if name.startswith('src/arkb/evaluation/'):continue
            if (root/name).read_bytes()!=subprocess.check_output(['git','show','75ba733:'+name],cwd=root):
                raise ValueError('Non-evaluation production source drift: '+name)
        tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only','75ba733','src'],text=True,cwd=root).splitlines())
        for path in (root/'src').rglob('*.py'):
            name=path.relative_to(root).as_posix()
            if name not in tracked and not name.startswith('src/arkb/evaluation/'):raise ValueError('Unexpected production addition: '+name)
        for entry in audit['changes']:
            if digest(root/entry['path'])!=entry['current_sha256']:raise ValueError('Audited evaluation source drift.')
        locations['status']='completed';locations['completed_at']=datetime.now(timezone.utc).isoformat()
        write_json(v/'artifact-locations.json',locations)
        meta.update(status='completed',completed_at=locations['completed_at'],
            validation_queries={'nfcorpus':323,'fiqa':648,'browsecomp-plus':830},
            decision='A: keep current chunk-level Hybrid fusion',production_behavior_changes=0)
        write_json(v/'completion.json',meta)
        readme=v/'README.md';text=readme.read_text()
        text=text.replace('Broader validation is in progress; do not interpret this directory as a completed\nPhase C release.',
            'Broader validation and preservation are complete. See `completion.json` and the engineering report.')
        text=text.replace('The engineering report is `docs/phase-c-report.md`; pending validation is marked explicitly.',
            'The completed engineering report is `docs/phase-c-report.md`.')
        readme.write_text(text)
        subprocess.run([sys.executable,str(root/'evaluation/audits/report_phase_c.py')],check=True,cwd=root)
        print('All Phase C validations, offline replays, snapshots and regression evidence verified.',flush=True)
    except BaseException as error:
        meta.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        meta['updated_at']=datetime.now(timezone.utc).isoformat();write_json(status,meta)


if __name__=='__main__':main()

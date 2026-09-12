"""Finish the two registered full validations when their production caches are ready.

A single local execution, not a scheduler. Stops on any failure, never retries a
retrieval capture or selects a policy. Large evidence stays on the approved disk.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from arkb.evaluation.external import digest,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--storage',type=Path,required=True)
    p.add_argument('--qdrant-url',default='http://127.0.0.1:6340');a=p.parse_args()
    root=Path(__file__).resolve().parents[2];versioned=root/'evaluation/phase-c/v1/validation'
    status_path=a.storage/'validation/completion-status.json'
    meta={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'jobs':{}}
    try:
        for ds,cache_name,run_name,empty in [('fiqa','fiqa-r2','fiqa-r2',True),
                                           ('browsecomp-plus','browsecomp-plus','browsecomp-plus',False)]:
            cache=a.storage/'cache'/cache_name;run=a.storage/'validation'/run_name
            if run.exists():raise FileExistsError('Do not repeat a validation capture: '+str(run))
            meta['jobs'][ds]={'status':'waiting_for_embedding_cache'};write_json(status_path,meta)
            print(ds,'waiting for complete cache',flush=True)
            while True:
                state=json.loads((cache/'status.json').read_text())
                if state['status']=='failed':raise RuntimeError(ds+' cache failed; inspect original status and evidence')
                if state['status']=='completed':break
                time.sleep(30)
            if digest(cache/'index.sqlite')!=state['cache_sha256']:raise ValueError('Cache changed after completion')
            meta['jobs'][ds]['status']='indexing_and_single_capture';write_json(status_path,meta)
            args=[sys.executable,str(root/'evaluation/experiments/validate_phase_c.py'),
                '--dataset',str(a.storage/'data'/ds),'--output',str(run),'--reuse-cache',str(cache/'index.sqlite'),
                '--qdrant-url',a.qdrant_url,*(['--omit-empty-documents'] if empty else [])]
            subprocess.run(args,check=True,cwd=root)
            subprocess.run([sys.executable,str(root/'evaluation/audits/replay_phase_c_validation.py'),
                '--dataset',str(a.storage/'data'/ds),'--run',str(run),'--output',str(versioned/(ds+'-replay.json'))],check=True,cwd=root)
            for name in ('summary','protocol','experiment','checksums'):
                shutil.copyfile(run/(name+'.json'),versioned/(ds+'-'+name+'.json'))
            shutil.copyfile(cache/'status.json',versioned/(ds+'-embedding-cache.json'))
            meta['jobs'][ds]={'status':'completed','run':str(run),'checksums_sha256':digest(run/'checksums.json')}
            write_json(status_path,meta)
            subprocess.run([sys.executable,str(root/'evaluation/audits/report_phase_c.py')],check=True,cwd=root)
            print(ds,'completed and replay-verified',flush=True)
        meta['status']='completed'
    except BaseException as error:
        meta.update(status='failed',error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        meta['updated_at']=datetime.now(timezone.utc).isoformat();write_json(status_path,meta)


if __name__=='__main__':main()

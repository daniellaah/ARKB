"""One retrieval-only validation capture, followed by frozen-leg fusion scoring."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
from arkb.evaluation.external import digest,load_external,write_json,rank_metrics,reference_metrics,verify_checksums
from arkb.evaluation.environment import runtime_environment
from arkb.retrieval.hybrid import rrf
from arkb.runtime import Runtime
from arkb.config import RuntimeConfig,RetrievalConfig
from arkb.knowledge.sqlite import SQLiteStorage
from freeze_phase_c import unique,emit
from run_p4 import verify_snapshot


def execute(a):
    verify_checksums(a.dataset)
    out=a.output;data=load_external(a.dataset);mapping=data.source_map();data.verify_materialized(a.dataset/'corpus')
    if digest(a.dataset/'checksums.json')!=digest(out/'input-checksums.json'):raise ValueError('Input checksum manifest drift.')
    decision=json.loads((out/'development-decision.json').read_text())
    if decision['selected_policy']!='C0':raise ValueError('This validation run was frozen for selected C0.')
    meta={'status':'indexing','started_at':datetime.now(timezone.utc).isoformat(),'dataset':data.name,
        'selected_policy':'C0','selected_alias':'C0','agent':False,'reranker':False,
        'dataset_manifest_sha256':digest(a.dataset/'manifest.json')}
    corpus_dir=a.dataset/'corpus'
    if a.omit_empty_documents:
        corpus_dir=out/'index-corpus'
        data.materialize(corpus_dir,omit_empty=True)
        data.verify_materialized(corpus_dir,omit_empty=True)
        empty=[d['id'] for d in data.corpus if not (d['title']+d['text']).strip()]
        meta['empty_document_exception']={'excluded_ids':empty,'indexed_documents':len(data.corpus)-len(empty),
            'full_corpus_and_qrels_preserved':True,'positive_labels_remain_in_denominator':True}
    write_json(out/'experiment.json',meta)
    config=RuntimeConfig(offline=True,tokenizer_cache=Path('.uv-cache/tokenizers').resolve(),qdrant_url=a.qdrant_url)
    try:
        with Runtime(config) as runtime:
            meta['environment_before']=runtime_environment(runtime)
            tags=lambda:{m.model:m.digest for m in runtime.model_client().list().models if m.model=='qwen3-embedding:0.6b'}
            meta['models_before']=tags();write_json(out/'experiment.json',meta)
            print(data.name,'indexing complete corpus',len(data.corpus),flush=True);start=perf_counter()
            build=runtime.index(db=out/'index.sqlite',vault_id=data.name,notes_dir=corpus_dir,
                chunking='recursive',chunk_size=512,chunk_overlap=64,batch_size=32,max_batch_tokens=8192)
            meta['index_setup_ms']=(perf_counter()-start)*1000;meta['index_report']=asdict(build)
            meta['status']='capturing_legs';write_json(out/'experiment.json',meta)
            print(data.name,'indexed',build.manifest.chunk_count,meta['index_setup_ms'],flush=True)
            with SQLiteStorage(out/'index.sqlite',read_only=True) as storage:
                meta['snapshot_before']=verify_snapshot(runtime,storage,build.manifest);write_json(out/'experiment.json',meta)
                engine=runtime.retrieval_engine(storage,build.manifest,modes=('bm25','semantic'),rerank=False,
                    exact=True,settings=RetrievalConfig(candidate_k=500))
                leg_hashes={name:hashlib.sha256() for name in ('bm25','semantic')}
                with gzip.open(out/'legs.jsonl.gz','wt',compresslevel=1) as f:
                    for i,(qid,query) in enumerate(data.queries.items()):
                        row={'dataset':data.name,'qid':qid,'query':query,'index_id':build.manifest.index_version,
                             'legs':{},'retrieval_ms':{}}
                        for name in ('bm25','semantic'):
                            start=perf_counter();response=engine.search(query,mode=name,top_k=500)
                            if response.index_id!=build.manifest.index_version:raise ValueError('Snapshot drift.')
                            row['retrieval_ms'][name]=(perf_counter()-start)*1000
                            row['legs'][name]=[{'rank':j+1,'hit':asdict(h)} for j,h in enumerate(response.results)]
                            leg_hashes[name].update((json.dumps({'qid':qid,'query':query,'results':row['legs'][name]},sort_keys=True,ensure_ascii=False)+'\n').encode())
                        emit(f,row)
                        if i%10==0:print(data.name,'captured',i+1,'/',len(data.queries),flush=True)
                meta['leg_sha256']={name:h.hexdigest() for name,h in leg_hashes.items()}
                meta['snapshot_after']=verify_snapshot(runtime,storage,build.manifest)
                if meta['snapshot_before']!=meta['snapshot_after']:raise ValueError('Snapshot drift.')
                meta['backend']=storage.build_metadata(build.manifest.index_version)['backend']
            meta['models_after']=tags();meta['environment_after']=runtime_environment(runtime)
            if meta['models_before']!=meta['models_after'] or meta['environment_before']!=meta['environment_after']:
                raise ValueError('Model/environment drift.')
        meta['status']='scoring_frozen_legs';meta['legs_sha256']=digest(out/'legs.jsonl.gz');write_json(out/'experiment.json',meta)
        score(a,data)
        data.verify_materialized(a.dataset/'corpus');load_external(a.dataset);verify_checksums(a.dataset)
        if digest(a.dataset/'checksums.json')!=digest(out/'input-checksums.json'):raise ValueError('Input checksum manifest drift.')
        if a.omit_empty_documents:data.verify_materialized(corpus_dir,omit_empty=True)
        meta['status']='completed';meta['index_sqlite_sha256']=digest(out/'index.sqlite')
    except BaseException as e:
        meta.update(status='failed',error={'type':type(e).__name__,'message':str(e)})
        raise
    finally:
        meta['finished_at']=datetime.now(timezone.utc).isoformat();write_json(out/'experiment.json',meta)
        write_json(out/'checksums.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()
            and p.name not in ('checksums.json','execution.log') and not p.name.endswith(('.pyc','-wal','-shm'))})


def score(a,data):
    from arkb.retrieval.models import SearchResult
    mapping=data.source_map();rows=[];goldpath=a.dataset/'gold-qrels.json'
    labels={'evidence':data.qrels} if data.name=='browsecomp-plus' else {'qrels':data.qrels}
    if goldpath.exists():labels['gold']=json.loads(goldpath.read_text())
    ks=(5,10,20,100,1000) if goldpath.exists() else (10,20,100)
    with gzip.open(a.output/'legs.jsonl.gz','rt') as source,gzip.open(a.output/'rows.jsonl.gz','wt',compresslevel=1) as target:
        for line in source:
            row=json.loads(line);qid=row['qid'];legs={name:[SearchResult(**r['hit']) for r in hits] for name,hits in row['legs'].items()}
            assert row['query']==data.queries[qid]
            start=perf_counter();fused=unique(rrf(legs,k=60),1000);fusion_ms=(perf_counter()-start)*1000
            for arm,hits in {**{n:unique(h,1000) for n,h in legs.items()},'C0':fused}.items():
                ranking=[mapping[h.source] for h in hits]
                metrics={label:rank_metrics(qrels[qid],ranking,ks=ks) for label,qrels in labels.items()}
                result={'qid':qid,'arm':arm,'ranking':ranking,'metrics':metrics,'returned_sources':len(ranking),
                    'fusion_ms':fusion_ms if arm=='C0' else None,
                    'candidate_chunks_processed':sum(map(len,legs.values())),
                    'representatives':[{'document_id':mapping[h.source],'source_id':h.source_id,'chunk_id':h.chunk_id,
                        'start_char':h.start_char,'end_char':h.end_char,'document_revision':h.metadata['document_revision'],
                        'score':h.score} for h in hits]}
                rows.append(result);emit(target,result)
    summary={'dataset':data.name,'corpus_count':len(data.corpus),'query_count':len(data.queries),
        'leg_depth':500,'primary_source_cutoff':100,'selected_policy':'C0','selected_identical_to_C0':True,
        'official_depth_note':'Recall@1000 uses the available full source ranking from fixed 500-chunk legs; actual depth is reported.',
        'arms':{},'paired_selected_minus_C0':{}}
    for arm in ('bm25','semantic','C0'):
        selected=[r for r in rows if r['arm']==arm]
        summary['arms'][arm]={'labels':{},'returned_sources':{'mean':float(np.mean([r['returned_sources'] for r in selected])),
            'min':min(r['returned_sources'] for r in selected),'max':max(r['returned_sources'] for r in selected)}}
        for label,qrels in labels.items():
            metrics={m:float(np.mean([r['metrics'][label][m] for r in selected])) for m in selected[0]['metrics'][label]}
            refs=reference_metrics(qrels,{r['qid']:r['ranking'] for r in selected},ks=ks)
            error=max(abs(value-r['metrics'][label][m]) for r in selected for m,value in refs[r['qid']].items())
            if error>1e-9:raise ValueError('Independent metric check failed.')
            summary['arms'][arm]['labels'][label]={'metrics':metrics,'reference_max_error':error}
            if arm=='C0':
                summary['paired_selected_minus_C0'][label]={m:{'mean_delta':0.,'ci95':[0.,0.],
                    'wins':0,'ties':len(selected),'losses':0} for m in metrics}
        if arm=='C0':
            times=[r['fusion_ms'] for r in selected]
            summary['arms'][arm]['fusion_ms']={'mean':float(np.mean(times)),'p50':float(np.median(times)),'p95':float(np.quantile(times,.95))}
        with (a.output/(arm+'.trec')).open('x') as trec:
            for row in selected:
                for rank,doc in enumerate(row['ranking'],1):trec.write(f'{row["qid"]} Q0 {doc} {rank} {len(row["ranking"])-rank+1} ARKB-{arm}\n')
    write_json(a.output/'summary.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qdrant-url',default='http://127.0.0.1:6340');p.add_argument('--execute',action='store_true');p.add_argument('--reuse-cache',type=Path)
    p.add_argument('--omit-empty-documents',action='store_true')
    a=p.parse_args();a.dataset=a.dataset.resolve();a.output=a.output.resolve()
    if a.execute:execute(a);return
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
    if a.reuse_cache:
        import sqlite3
        with sqlite3.connect(a.reuse_cache.resolve().as_uri()+'?mode=ro',uri=True) as source:
            with sqlite3.connect(a.output/'index.sqlite') as target:source.backup(target)
    shutil.copytree(root/'src',a.output/'measured-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','uv.lock'):shutil.copyfile(root/name,a.output/'measured-source'/name)
    for name in ('validate_phase_c.py','freeze_phase_c.py','run_p4.py','run_phase_c.py'):
        shutil.copyfile(Path(__file__).with_name(name),a.output/name)
    shutil.copyfile(root/'evaluation/phase-c/v1/development-decision.json',a.output/'development-decision.json')
    shutil.copyfile(a.dataset/'manifest.json',a.output/'data-manifest.json')
    shutil.copyfile(a.dataset/'checksums.json',a.output/'input-checksums.json')
    write_json(a.output/'protocol.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'input_checksums_sha256':digest(a.output/'input-checksums.json'),
        'dataset_manifest_sha256':digest(a.dataset/'manifest.json'),'development_decision_sha256':digest(a.output/'development-decision.json'),
        'reused_embedding_cache_sha256':digest(a.reuse_cache) if a.reuse_cache else None,
        'omit_empty_documents':a.omit_empty_documents,
        'chunk_size':512,'chunk_overlap':64,'leg_depth':500,'rrf_k':60,'rerank':False,'agent':False,'exact':True,
        'query_ids':json.loads((a.dataset/'manifest.json').read_text())['query_ids'],'qdrant_url':a.qdrant_url,
        'source_files':{p.relative_to(a.output/'measured-source').as_posix():digest(p) for p in (a.output/'measured-source').rglob('*') if p.is_file()}})
    env={**os.environ,'PYTHONPATH':str(a.output/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    with (a.output/'execution.log').open('x') as log:
        result=subprocess.run([sys.executable,str(a.output/'validate_phase_c.py'),'--execute','--dataset',str(a.dataset),
            '--output',str(a.output),'--qdrant-url',a.qdrant_url,
            *(['--omit-empty-documents'] if a.omit_empty_documents else [])],env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'output':str(a.output),'exit_code':result.returncode}));raise SystemExit(result.returncode)

if __name__=='__main__':main()

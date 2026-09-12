"""Precompute immutable production embedding-cache entries, without publishing an index.

Operational staging for the frozen full-corpus validation. Uses the exact same
chunker, input preparation, token limits, model and SQLite cache as build_index.
No retrieval or scoring occurs; a later normal build verifies/reuses these keys.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter

from arkb.runtime import Runtime
from arkb.config import RuntimeConfig
from arkb.knowledge.documents import scan_notes
from arkb.knowledge.chunking import chunk_notes
from arkb.knowledge.embeddings import (count_tokens,resolve_embedding_spec,prepare_document,
    validate_input_tokens,iter_embedding_batches,tokenizer_fingerprint)
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.evaluation.external import digest,write_json


def load_prepared_inputs(plan, *, dataset, tokenizer, spec):
    """Verify a completed parallel preparation plan before opening a cache writer."""
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((plan / 'manifest.json').read_text())
    identity = json.loads((plan / 'identity.json').read_text())
    original = json.loads((plan / 'original-serial-checkpoint.json').read_text())
    source_files = ('src/arkb/knowledge/chunking.py', 'src/arkb/knowledge/documents.py',
                    'src/arkb/knowledge/embeddings.py', 'src/arkb/knowledge/models.py')
    if (manifest.get('schema') != 'arkb-embedding-input-plan-v1'
            or manifest.get('status') != 'completed'
            or manifest['embedding_spec'] != asdict(spec)
            or manifest['tokenizer'] != tokenizer_fingerprint(tokenizer)
            or manifest['dataset_manifest_sha256'] != digest(dataset / 'manifest.json')
            or manifest['chunk_size'] != 512 or manifest['chunk_overlap'] != 64
            or any(manifest.get(key) != value for key, value in identity.items())
            or manifest['expected_checkpoint_sha256'] != digest(plan / 'original-serial-checkpoint.json')
            or manifest['script_sha256'] != digest(plan / 'prepare_phase_c_inputs.py')
            or manifest['script_sha256'] != digest(root / 'evaluation/experiments/prepare_phase_c_inputs.py')
            or manifest['source_files'] != {name: digest(root / name) for name in source_files}
            or manifest['document_plans_sha256'] != digest(plan / 'document-plans.jsonl')):
        raise ValueError('Prepared input plan identity/checksum mismatch.')
    for key in ('documents', 'chunks', 'unique_inputs', 'input_sha256', 'tokenizer',
                'dataset_manifest_sha256', 'embedding_spec', 'chunk_size', 'chunk_overlap'):
        if manifest[key] != original[key]:
            raise ValueError('Prepared inputs differ from original serial checkpoint: ' + key)
    if original['batch_size'] != 32 or original['max_batch_tokens'] != 8192:
        raise ValueError('Original request limits differ from this cache helper.')
    unique, checksum = {}, hashlib.sha256()
    with (plan / 'inputs.jsonl').open('rb') as stream:
        for line in stream:
            checksum.update(line)
            text, tokens = json.loads(line)
            if (not isinstance(text, str) or not text.strip() or text in unique
                    or type(tokens) is not int or not 0 < tokens <= 8192):
                raise ValueError('Invalid or duplicate prepared embedding input.')
            unique[text] = tokens
    if len(unique) != manifest['unique_inputs'] or checksum.hexdigest() != manifest['inputs_file_sha256']:
        raise ValueError('Prepared input file checksum/count mismatch.')
    return manifest, unique


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--omit-empty-documents',action='store_true')
    p.add_argument('--prepared-plan',type=Path,help='Reuse a verified original-equivalent input plan; no model/configuration change.')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.prepared_plan and a.omit_empty_documents:
        raise ValueError('Prepared plans require the complete original corpus.')
    meta={'status':'preparing','started_at':datetime.now(timezone.utc).isoformat(),
        'dataset_manifest_sha256':digest(a.dataset/'manifest.json'),'index_published':False,
        'script_sha256':digest(__file__),'batch_size':32,'max_batch_tokens':8192,'chunk_size':512,'chunk_overlap':64}
    write_json(a.output/'status.json',meta);start=perf_counter()
    try:
        with Runtime(RuntimeConfig(offline=True,tokenizer_cache=Path('.uv-cache/tokenizers').resolve())) as runtime:
            tokenizer=runtime.tokenizer();client=runtime.model_client()
            spec=resolve_embedding_spec(client,'qwen3-embedding:0.6b',context_length=8192)
            meta['embedding_spec']=asdict(spec);meta['tokenizer']=tokenizer_fingerprint(tokenizer)
            if a.prepared_plan:
                print(a.dataset.name,'verifying saved input plan',flush=True)
                plan,unique=load_prepared_inputs(a.prepared_plan,dataset=a.dataset,tokenizer=tokenizer,spec=spec)
                meta.update({key:plan[key] for key in ('documents','chunks','unique_inputs','input_sha256')})
                meta['prepared_plan']={'directory':str(a.prepared_plan.resolve()),
                    'manifest_sha256':digest(a.prepared_plan/'manifest.json'),
                    'original_checkpoint_sha256':plan['expected_checkpoint_sha256'],
                    'inputs_file_sha256':plan['inputs_file_sha256']}
                meta['empty_document_exception']={'enabled':False,'excluded_sources':[],'raw_documents':plan['documents']}
            else:
                print(a.dataset.name,'reading complete materialized corpus',flush=True)
                notes=scan_notes(a.dataset/'corpus')
                empty=[n.source for n in notes if not (n.title+n.content).strip()]
                meta['empty_document_exception']={'enabled':a.omit_empty_documents,
                    'excluded_sources':empty if a.omit_empty_documents else [],'raw_documents':len(notes)}
                if a.omit_empty_documents:notes=[n for n in notes if (n.title+n.content).strip()]
                chunks=chunk_notes(notes,count_tokens=partial(count_tokens,tokenizer=tokenizer),chunk_size=512,chunk_overlap=64)
                unique={}
                input_hash=hashlib.sha256()
                for chunk in chunks:
                    text=prepare_document(chunk,document_template=spec.document_template)
                    input_hash.update((json.dumps(text,ensure_ascii=False)+'\n').encode())
                    if text not in unique:unique[text]=validate_input_tokens(text,tokenizer=tokenizer,max_tokens=8192)
                meta.update(documents=len(notes),chunks=len(chunks),unique_inputs=len(unique),input_sha256=input_hash.hexdigest())
                del chunks,notes
            print(a.dataset.name,'prepared',meta['documents'],meta['chunks'],meta['unique_inputs'],flush=True)
            with SQLiteStorage(a.output/'index.sqlite') as storage,storage.writer_lock():
                missing=[text for text in unique if storage.get_embedding(spec,text) is None]
                meta.update(status='embedding',cached_inputs=len(unique)-len(missing),completed_new=0)
                write_json(a.output/'status.json',meta)
                for offset,vectors in iter_embedding_batches(missing,client=client,model=spec.model,batch_size=32,
                    token_counts=[unique[text] for text in missing],max_batch_tokens=8192,dimensions=spec.dimensions,
                    dtype=spec.dtype,normalization=spec.normalization,max_retries=2,context_length=8192):
                    storage.put_embeddings(spec,missing[offset:offset+len(vectors)],vectors)
                    meta['completed_new']=offset+len(vectors)
                    if offset//1000!=(offset+len(vectors))//1000:
                        meta['elapsed_seconds']=perf_counter()-start;write_json(a.output/'status.json',meta)
                        print(a.dataset.name,'cached',meta['completed_new'],'/',len(missing),flush=True)
            if resolve_embedding_spec(client,spec.model,context_length=8192)!=spec:raise ValueError('Embedding identity drift.')
        meta.update(status='completed',elapsed_seconds=perf_counter()-start,cache_sha256=digest(a.output/'index.sqlite'))
    except BaseException as e:
        meta.update(status='failed',error={'type':type(e).__name__,'message':str(e)})
        raise
    finally:
        meta['finished_at']=datetime.now(timezone.utc).isoformat();write_json(a.output/'status.json',meta)
    print(a.dataset.name,meta['status'],meta.get('elapsed_seconds'),flush=True)

if __name__=='__main__':main()

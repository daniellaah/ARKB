"""Compare Qdrant upsert requests and temporary allocation against Phase B.

Uses a deterministic synthetic transport fixture, no server, model or benchmark
labels. Only generated wire requests and observed traced allocations are compared.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
import tracemalloc
from types import SimpleNamespace
import numpy as np
from qdrant_client import models
from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.models import ChunkRecord,EmbeddingSpec,Note
from arkb.evaluation.external import digest,write_json


class Capture:
    def __init__(self,spec,identity):
        self.spec=spec;self.identity=identity;self.batches=[];self.wire=hashlib.sha256()
    def get_collection(self,name):
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=models.VectorParams(
            size=self.spec.dimensions,distance=models.Distance.COSINE)),metadata=self.identity))
    def upsert(self,*,collection_name,points,wait):
        self.batches.append(len(points))
        self.wire.update((collection_name+':'+str(wait)+'\n').encode())
        for point in points:self.wire.update((point.model_dump_json()+'\n').encode())


def run(module,records,vectors,spec):
    client=Capture(spec,module.qdrant_identity(spec,'allocation-audit'))
    index=module.QdrantIndex(client,'allocation-audit',spec,vault_id='allocation-audit')
    tracemalloc.start();start=perf_counter()
    index.upsert(records,vectors)
    elapsed=perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    invalid=vectors.copy();invalid[-1]=0
    failure_client=Capture(spec,module.qdrant_identity(spec,'allocation-audit'))
    failure_index=module.QdrantIndex(failure_client,'allocation-audit',spec,vault_id='allocation-audit')
    try:failure_index.upsert(records,invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid tail vector accepted.')
    assert not failure_client.batches,'Full input validation must precede every write.'
    return {'batches':client.batches,'wire_sha256':client.wire.hexdigest(),'traced_peak_bytes':peak,
        'traced_retained_bytes':current,'audit_elapsed_seconds':elapsed,'invalid_tail_rejected_before_any_write':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    baseline=subprocess.check_output(['git','show','75ba733:src/arkb/knowledge/qdrant.py'],cwd=root)
    from arkb.knowledge import qdrant as current
    n=4097;spec=EmbeddingSpec(model='fixture',model_revision='fixed',dimensions=1024,
        document_template='title-body-v1',normalization='none')
    notes=[Note(title='Fixture',content='verbatim evidence',source=f'{i}.md') for i in range(n)]
    records=[ChunkRecord.from_note(c,note=n,vault_id='allocation-audit') for n,c in zip(notes,whole_note_chunks(notes))]
    vectors=np.zeros((n,spec.dimensions));vectors[:,0]=1
    with tempfile.TemporaryDirectory(prefix='arkb-upsert-audit-') as directory:
        path=Path(directory)/'baseline.py';path.write_bytes(baseline)
        definition=importlib.util.spec_from_file_location('phase_b_qdrant',path)
        old=importlib.util.module_from_spec(definition);definition.loader.exec_module(old)
        results={'baseline':run(old,records,vectors,spec),'current':run(current,records,vectors,spec)}
    assert results['baseline']['wire_sha256']==results['current']['wire_sha256']
    assert results['baseline']['batches']==results['current']['batches']==[128]*32+[1]
    results.update(status='passed',records=n,dimensions=spec.dimensions,
        baseline_source_sha256=hashlib.sha256(baseline).hexdigest(),current_source_sha256=digest(current.__file__),
        protocol='All records/vectors validated first. Same ordered batches, point IDs, vector values, payloads, collection, and wait=True.',
        measurement='tracemalloc during public upsert with a hashing transport; input matrix and records preexist. These timings include tracing/serialization and are not production latency.',
        model_calls=0,server_calls=0)
    write_json(a.output,results);print(json.dumps(results,indent=2))


if __name__=='__main__':main()

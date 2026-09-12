"""Preserve a completed validation index and a verified external Qdrant snapshot."""
import argparse
from contextlib import closing
from datetime import datetime,timezone
import json
from pathlib import Path
from urllib.parse import quote
import httpx
from qdrant_client import QdrantClient
from arkb.evaluation.external import digest,verify_checksums,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    verify_checksums(a.run);verify_checksums(a.dataset)
    meta=json.loads((a.run/'experiment.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
    if meta['status']!='completed':raise ValueError('Cannot preserve an incomplete validation as completed.')
    collection=meta['backend']['collection'];url=protocol['qdrant_url']
    with closing(QdrantClient(url=url,timeout=300)) as client:
        snapshot=client.create_snapshot(collection_name=collection,wait=True)
        if snapshot is None:raise ValueError('Snapshot creation returned no metadata.')
        target=a.output/snapshot.name
        with httpx.stream('GET',f'{url}/collections/{quote(collection,safe="")}/snapshots/{quote(snapshot.name,safe="")}',timeout=300) as response:
            response.raise_for_status()
            with target.open('xb') as f:
                for chunk in response.iter_bytes():f.write(chunk)
        checksum=digest(target)
        if snapshot.checksum and checksum!=snapshot.checksum:raise ValueError('Downloaded Qdrant snapshot checksum mismatch.')
        if target.stat().st_size!=snapshot.size:raise ValueError('Downloaded Qdrant snapshot size mismatch.')
        manifest={'dataset':meta['dataset'],'preserved_at':datetime.now(timezone.utc).isoformat(),
            'run':str(a.run.resolve()),'dataset_directory':str(a.dataset.resolve()),
            'run_checksums_sha256':digest(a.run/'checksums.json'),'dataset_checksums_sha256':digest(a.dataset/'checksums.json'),
            'sqlite':str((a.run/'index.sqlite').resolve()),'sqlite_sha256':meta['index_sqlite_sha256'],
            'index_manifest':meta['index_report']['manifest'],'backend':meta['backend'],'qdrant_url':url,
            'qdrant_snapshot':str(target.resolve()),'snapshot_sha256':checksum,'snapshot_metadata':snapshot.model_dump(mode='json'),
            'live_corpus':str((a.run/'index-corpus' if protocol.get('omit_empty_documents') else a.dataset/'corpus').resolve()),
            'query_ids':protocol['query_ids'],'retrieval_configuration':{'leg_depth':500,'exact':True,'rrf_k':60,'rerank':False},
            'phase_e_handoff':'Use the same dataset, query IDs, source directory and SQLite snapshot. If migrating Qdrant, restore this collection snapshot under the recorded collection name and point runtime at the restored server. Do not rebuild or redefine corpus scope.'}
        write_json(a.output/'manifest.json',manifest);write_json(a.manifest,manifest)
        # Only remove this newly created server-side snapshot after external verification.
        # Keep the live collection, Docker volume, external file and SQLite unchanged.
        client.delete_snapshot(collection_name=collection,snapshot_name=snapshot.name,wait=True)
    print(meta['dataset'],'Qdrant snapshot preserved and hash-verified',snapshot.size,flush=True)


if __name__=='__main__':main()

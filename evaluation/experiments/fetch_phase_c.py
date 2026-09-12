"""Fetch pinned official validation inputs; hash-verify every file, no remote code."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import httpx
from arkb.evaluation.external import digest,write_json


def fetch(url,path,expected):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        checksum=digest(path)
        if expected.get('sha256') and checksum==expected['sha256'] and path.stat().st_size==expected['bytes']:
            return checksum
    temporary=path.with_suffix(path.suffix+'.partial')
    for attempt in range(3):
        try:
            with httpx.stream('GET',url,follow_redirects=True,timeout=120) as response,temporary.open('wb') as f:
                response.raise_for_status()
                for chunk in response.iter_bytes(1024*1024):f.write(chunk)
            checksum=digest(temporary)
            if temporary.stat().st_size!=expected['bytes']:raise ValueError('Download size changed.')
            if expected.get('sha256') and checksum!=expected['sha256']:raise ValueError('Download SHA256 changed.')
            if expected.get('git_blob'):
                raw=temporary.read_bytes();actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
                if actual!=expected['git_blob']:raise ValueError('Download Git blob identity changed.')
            temporary.replace(path);return checksum
        except Exception:
            if attempt==2:raise
            time.sleep(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    revisions=json.loads(Path('evaluation/phase-c/v1/acquisition-revisions.json').read_text())
    lock={}
    for repo,info in revisions.items():
        name=repo.split('/')[1]
        for item in info['siblings']:
            file=item['rfilename']
            if not(file=='README.md' or file.endswith('.parquet') or file=='test.tsv'):continue
            relative=name+'/'+file
            lock[relative]={'url':f'https://huggingface.co/datasets/{repo}/resolve/{info["sha"]}/{file}',
                'revision':info['sha'],'repository':repo,'bytes':item['size'],
                'sha256':item.get('lfs',{}).get('sha256'),'git_blob':item.get('blobId') if not item.get('lfs') else None}
    official=json.loads(Path('evaluation/phase-c/v1/browsecomp-official-revision.json').read_text())
    for item in official['files']:
        name='browsecomp-official/'+item['path']
        lock[name]={'url':f'https://raw.githubusercontent.com/texttron/BrowseComp-Plus/{official["revision"]}/{item["path"]}',
            'revision':official['revision'],'repository':'texttron/BrowseComp-Plus','bytes':item['size'],'git_blob':item['sha']}
    write_json(a.output/'download-lock.json',lock)
    manifest_path=a.output/'downloads.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files':{}}
    for name,expected in lock.items():
        if name in manifest['files'] and digest(a.output/name)==manifest['files'][name]['sha256']:continue
        checksum=fetch(expected['url'],a.output/name,expected)
        manifest['files'][name]={**expected,'sha256':checksum};write_json(manifest_path,manifest)
        print(name,expected['bytes'],flush=True)

if __name__=='__main__':main()

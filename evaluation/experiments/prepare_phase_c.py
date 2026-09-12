"""Normalize full official validation corpora with separate scoring artifacts."""
import argparse
import json
from pathlib import Path
from arkb.evaluation.external import import_beir,write_json,digest
from arkb.evaluation.browsecomp import import_browsecomp


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',choices=('nfcorpus','fiqa','browsecomp-plus'),required=True);a=p.parse_args()
    if a.dataset=='browsecomp-plus':data,gold,urls=import_browsecomp(a.inputs)
    else:
        data=import_beir(a.inputs,a.dataset)
        expected={'nfcorpus':(3633,323),'fiqa':(57638,648)}[a.dataset]
        if (len(data.corpus),len(data.queries))!=expected:raise ValueError('Incomplete BEIR corpus/query set.')
    data.provenance['exposure']='broader-validation-after-frozen-development-decision'
    data.provenance['development_decision_sha256']=digest('evaluation/phase-c/v1/development-decision.json')
    data.save(a.output,materialize=True)
    if a.dataset=='browsecomp-plus':
        write_json(a.output/'gold-qrels.json',gold);write_json(a.output/'official-urls.json',urls)
    write_json(a.output/'checksums.json',{p.name:digest(p) for p in a.output.iterdir() if p.is_file()})
    print(json.dumps({'dataset':a.dataset,'corpus':len(data.corpus),'queries':len(data.queries),'output':str(a.output)}),flush=True)

if __name__=='__main__':main()

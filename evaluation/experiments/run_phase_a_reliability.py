"""One frozen, eight-case protocol comparison; never searches for a better prompt.

Register after deterministic checks, then execute old and new source snapshots
in separate processes. Existing P4 corpora and published indexes are read-only.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 'ea6e395'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def hashes(directory):
    return {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('-wal','-shm'))}


def register(output):
    from arkb.evaluation.external import load_external
    from arkb.knowledge.sqlite import SQLiteStorage
    output.mkdir(parents=True, exist_ok=False)
    results = ROOT/'evaluation/results'
    samples = json.loads((results/'p4-inputs/official/bright-pro/agentic_sample_ids.json').read_text())
    cases = []
    for domain in ('stackoverflow','robotics'):
        dataset = results/'p4-data'/f'bright-{domain}'
        data = load_external(dataset)
        for qid in samples['tasks'][domain][:2]:
            cases.append({'key': f'bright-{domain}:{qid}', 'track': 'bright', 'domain': domain,
                'id': qid, 'query': data.queries[qid], 'dataset': str(dataset),
                'dataset_manifest_sha256': digest(dataset/'manifest.json'),
                'directory': str(dataset/'corpus'),
                'db': str(results/f'p4-bright-{domain}-indexed/index.sqlite'), 'vault': data.name})
    selected = json.loads((results/'p4-data/musique-selected.json').read_text())['rows'][:4]
    write(output/'musique-gold.json', selected)
    for i, row in enumerate(selected):
        cases.append({'key': f'musique:{i}:{row["id"]}', 'track': 'musique', 'id': row['id'],
            'context_index': i, 'query': row['question'],
            'directory': str(results/f'p4-musique-agent-v3/contexts/{i}'),
            'db': str(results/f'p4-musique-agent-v3/context-indexes/{i}.sqlite'), 'vault': 'p4-musique'})
    for case in cases:
        with SQLiteStorage(Path(case['db']), read_only=True) as storage:
            manifest = storage.active_manifest(case['vault'])
            backend = storage.build_metadata(manifest.index_version)['backend']
            if Path(backend['source_scope']).resolve() != Path(case['directory']).resolve():
                raise ValueError('Snapshot source scope mismatch.')
            case.update(index_manifest=asdict(manifest), sqlite_sha256=digest(Path(case['db'])))
    write(output/'cases.json', cases)
    # Only source files: do not switch or mutate the user's checkout.
    for arm in ('old','new'):
        source = output/arm/'measured-source'
        source.mkdir(parents=True)
        if arm == 'old':
            paths = subprocess.check_output(['git','ls-tree','-r','--name-only',BASELINE,'src'],cwd=ROOT,text=True).splitlines()
            for relative in paths:
                destination = source/relative; destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(subprocess.check_output(['git','show',f'{BASELINE}:{relative}'],cwd=ROOT))
        else:
            shutil.copytree(ROOT/'src',source/'src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        write(output/arm/'source-hashes.json', hashes(source))
    protocol = {'schema':'arkb-phase-a-reliability-v1','registered_at':datetime.now(timezone.utc).isoformat(),
        'baseline_commit':BASELINE,'candidate_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'selection':'First two registered SO cases, first two Robotics cases, first two MuSiQue pairs; no score-based selection.',
        'trials':1,'cases_per_arm':8,'model':'qwen3.5:4b','think':True,'temperature':0,'max_turns':8,
        'budget':{'max_tool_calls':12,'max_query_calls':10,'max_read_calls':6,'max_evidence_tokens':4000,'max_elapsed_ms':120000},
        'harness_deadline_seconds':180,'old_query_suffix':'Existing MuSiQue OUTPUT_INSTRUCTION, unchanged from P4.',
        'new_query_suffix':'None; production finish/constrained schema maps through canonical_prediction.',
        'confounds':['Necessary tool schemas, evidence serialization and finalization instructions differ.',
                     'The old MuSiQue prompt requests a short answer; the new canonical answer may contain explanations. F1 is diagnostic.',
                     'One trial per case with temperature 0 is not a reproducibility or statistical-significance claim.'],
        'cases_sha256':digest(output/'cases.json'),'musique_gold_sha256':digest(output/'musique-gold.json'),
        'release_eligible':False}
    write(output/'protocol.json',protocol)


def execute(output, arm, qdrant_url, *, resume=False):
    import arkb
    from arkb.runtime import Runtime
    from arkb.config import RuntimeConfig
    from arkb.knowledge.sqlite import SQLiteStorage
    from arkb.agent.observation import AgentBudget
    from arkb.evaluation.environment import runtime_environment
    from arkb.evaluation.deadline import evaluation_deadline
    from arkb.evaluation.external import load_external, ExternalDataset
    from arkb.evaluation.multihop import OUTPUT_INSTRUCTION, parse_prediction
    if not Path(arkb.__file__).resolve().is_relative_to(output/arm/'measured-source/src'):
        raise ValueError('Wrong measured source import.')
    protocol = json.loads((output/'protocol.json').read_text())
    if digest(output/'cases.json') != protocol['cases_sha256']: raise ValueError('Case registration changed.')
    cases = json.loads((output/'cases.json').read_text())
    gold = json.loads((output/'musique-gold.json').read_text())
    if hashes(output/arm/'measured-source') != json.loads((output/arm/'source-hashes.json').read_text()):
        raise ValueError('Measured source changed.')
    existing=[]
    if resume:
        existing=[json.loads(line) for line in (output/arm/'rows.jsonl').read_text().splitlines()]
        if [r['key'] for r in existing] != [c['key'] for c in cases[:len(existing)]]:
            raise ValueError('Resume rows do not match the registered prefix.')
    metadata = {'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'arm':arm,
                'resumed_after_rows':len(existing)}
    write(output/arm/'experiment.json',metadata)
    with Runtime(RuntimeConfig(offline=True,tokenizer_cache=ROOT/'.uv-cache/tokenizers',qdrant_url=qdrant_url)) as runtime:
        metadata['environment'] = runtime_environment(runtime)
        models = lambda: {m.model:m.digest for m in runtime.model_client().list().models
                          if m.model in ('qwen3.5:4b','qwen3-embedding:0.6b')}
        metadata['models_before'] = models()
        definition = runtime.model_client().show(protocol['model']).model_dump(mode='json')
        metadata['model_definition'] = {k:definition.get(k) for k in ('parameters','capabilities','model_info')}
        metadata['template_sha256'] = hashlib.sha256((definition.get('template') or '').encode()).hexdigest()
        (output/arm/'reference-tokenizer.json').write_text(runtime.tokenizer().to_str())
        write(output/arm/'experiment.json',metadata)
        with (output/arm/'rows.jsonl').open('a' if resume else 'x') as stream:
            # Bright preparations are shared within an arm, as in P4. MuSiQue
            # contexts retain independent snapshots and never share candidate pools.
            cached = {}
            with ExitStack() as resources:
                for case in cases[len(existing):]:
                    key = case['db']
                    prep = perf_counter()
                    if key not in cached:
                        storage = resources.enter_context(SQLiteStorage(Path(key),read_only=True))
                        manifest = storage.active_manifest(case['vault'])
                        if asdict(manifest) != case['index_manifest']: raise ValueError('Snapshot identity changed.')
                        engine = runtime.retrieval_engine(storage,manifest,modes=('bm25','semantic'),exact=True)
                        tools = runtime.agent_tools(engine=engine,directory=Path(case['directory']),vault_id=case['vault'])
                        cached[key] = tools
                    else: tools = cached[key]
                    if case['track'] == 'bright':
                        dataset = load_external(Path(case['dataset'])); source_map = dataset.source_map()
                        if digest(Path(case['dataset'])/'manifest.json') != case['dataset_manifest_sha256']:
                            raise ValueError('Dataset registration changed.')
                        qrels, aspects = dataset.qrels[case['id']], dataset.aspects.get(case['id'],[])
                    else:
                        row = gold[case['context_index']]
                        dataset = ExternalDataset(f'musique-{case["context_index"]}',[{'id':str(p['idx']),'title':p['title'],'text':p['paragraph_text']}
                            for p in row['paragraphs']],{'q':case['query']},{'q':{}},{},{})
                        source_map = {s:int(i) for s,i in dataset.source_map().items()}
                        qrels = {p['idx']:1 for p in row['paragraphs'] if p['is_supporting']}; aspects = []
                    dataset.verify_materialized(Path(case['directory']))
                    preparation_ms = (perf_counter()-prep)*1000
                    prompt = case['query'] + (OUTPUT_INSTRUCTION if arm=='old' and case['track']=='musique' else '')
                    observer = runtime.agent_observer(budget=AgentBudget(**protocol['budget']))
                    start=perf_counter(); result=None; error=None
                    try:
                        with evaluation_deadline(protocol['harness_deadline_seconds']):
                            result = runtime.run_agent(prompt,tools=tools,model=protocol['model'],max_turns=8,think=True,observer=observer)
                    except Exception as exc:
                        error={'type':type(exc).__name__,'message':str(exc)}; result=getattr(exc,'agent_result',None)
                    elapsed = (perf_counter()-start)*1000
                    if arm=='new' and result is not None and result.final.status=='error':
                        error=result.final.error or {'type':'AgentFailure','message':result.final.termination_reason}
                    if case['track']=='musique':
                        if arm=='old': prediction,parse_error=parse_prediction(result.response if result else '',source_map,stopped=result.stop_reason if result else 'error')
                        else:
                            from arkb.evaluation.multihop import canonical_prediction
                            prediction,parse_error=canonical_prediction(result.final if result else None,source_map)
                    else: prediction=parse_error=None
                    record={'key':case['key'],'track':case['track'],'id':case['id'],'source_map':source_map,
                        'qrels':qrels,'aspects':aspects,'preparation_ms':preparation_ms,'elapsed_ms':elapsed,
                        'error':error,'stop_reason':result.stop_reason if result else 'error',
                        'result':asdict(result) if result else None,'prediction':prediction,'parse_error':parse_error}
                    # Retain only observed source mappings for large Bright corpora.
                    if case['track']=='bright' and result is not None:
                        seen=set()
                        for event in result.observation['tools']:
                            raw=event.get('raw_result') or {}
                            for hit in ([raw.get('result')] if event['name']=='read' else raw.get('results',[])):
                                if isinstance(hit,dict): seen.add(hit['source'])
                        record['source_map']={s:d for s,d in source_map.items() if s in seen}
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()
                    print(arm,case['key'],record['stop_reason'],round(elapsed/1000,2),flush=True)
                    if digest(Path(case['db'])) != case['sqlite_sha256']: raise ValueError('Read-only snapshot bytes changed.')
        metadata['models_after']=models()
        if metadata['models_after']!=metadata['models_before']: raise ValueError('Model identity changed.')
    metadata.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat())
    write(output/arm/'experiment.json',metadata)
    write(output/arm/'checksums.json',hashes(output/arm))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--arm',choices=('old','new'))
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--qdrant-url',default='http://127.0.0.1:32776')
    args=parser.parse_args(); output=args.output.resolve()
    if args.register: register(output); return
    if not args.arm: parser.error('--arm is required for execution')
    if args.child: execute(output,args.arm,args.qdrant_url,resume=args.resume); return
    env={**os.environ,'PYTHONPATH':str(output/args.arm/'measured-source/src'),
         'PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--output',str(output),
                    '--arm',args.arm,'--child','--qdrant-url',args.qdrant_url, *(['--resume'] if args.resume else [])],env=env,check=True)


if __name__=='__main__': main()

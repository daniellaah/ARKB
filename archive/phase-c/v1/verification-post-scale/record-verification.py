from pathlib import Path
import json,shutil,xml.etree.ElementTree as E
from arkb.evaluation.external import write_json,digest,verify_checksums
root=Path.cwd();logs=root/'evaluation/results/phase-c-v1';out=root/'evaluation/phase-c/v1/verification-post-scale';out.mkdir(exist_ok=True)
normal=list(E.parse(logs/'normal-post-scale.xml').iter('testcase'));service=list(E.parse(logs/'service-post-scale.xml').iter('testcase'))
first_service=service
retry=list(E.parse(logs/'service-native-retry.xml').iter('testcase'))
def identity(case):return (case.get('classname'),case.get('name'))
def passed(case):return not any(case.find(k) is not None for k in ('failure','error','skipped'))
initial={identity(c):c for c in first_service}
assert len(initial)==len(first_service)==60
assert len(retry)==1 and not passed(initial[identity(retry[0])]) and passed(retry[0])
final_service=initial|{identity(c):c for c in retry}
service=list(final_service.values())
def stats(cases):
 return {'cases':len(cases),'passed':sum(not any(c.find(k) is not None for k in ('failure','error','skipped')) for c in cases),
         **{name:sum(c.find(tag) is not None for c in cases) for name,tag in [('failed','failure'),('errors','error'),('skipped','skipped')]}}
def selected(prefixes,cases=normal):return [c for c in cases if c.get('classname').startswith(tuple(prefixes))]
phasea=['tests.agent.','tests.interfaces.test_cli','tests.knowledge.test_documents','tests.retrieval.test_exact','tests.test_runtime']
phaseb=['tests.retrieval.test_rerank','tests.retrieval.test_qwen_rerank','tests.retrieval.test_qwen_inputs']
new=selected(['tests.evaluation.test_source_fusion','tests.evaluation.test_public_benchmarks'])
fresh=json.loads((logs/'freshness-post-scale/results.json').read_text());verify_checksums(logs/'freshness-post-scale')
report={'schema':'arkb-phase-c-verification-v1','groups_overlap':True,
 'deterministic':stats(normal),'service_integration':stats(service),'evaluation_regression':stats(selected(['tests.evaluation.'])),
 'phase_a_regression':stats(selected(phasea)),'phase_b_reranker_focused':stats(selected(phaseb)),
 'phase_b_reranker_integration':stats(selected(['tests.retrieval.integration.test_qwen_reranker_model'],service)),
 'new_phase_c_deterministic':stats(new),'new_case_ids':[c.get('classname')+'::'+c.get('name') for c in new],
 'freshness':{'passed':fresh['passed'],'failed':fresh['failed'],'checks':fresh['checks']}}
fixture=logs/'reranker-native-fixture'
manifest=json.loads((fixture/'manifest.json').read_text())
assert digest(manifest['original_sqlite'])==manifest['original_sqlite_sha256_before_and_after']
assert digest(fixture/'index.sqlite')==manifest['copied_sqlite_sha256']
report['service_execution']={
 'counting':'60 unique (classname, name) cases, final outcome after the single documented infrastructure retry. This is not one all-green invocation.',
 'attempts':61,'initial':stats(first_service),'retry':stats(retry),
 'retry_case':'::'.join(identity(retry[0])),
 'reason':'The original host-bind Qdrant 32776 timed out reading collection metadata during the second filtered semantic search. Earlier rerank and provenance assertions had passed.',
 'resolution':'Retried only the failed case against an isolated native Qdrant 6340 collection restored from the exact saved SQLite snapshot. Every stored vector/payload verified; original SQLite hash unchanged.',
 'fixture_manifest':'reranker-native-fixture/manifest.json',
 'raw_attempts':['service-post-scale.xml','service-native-retry.xml']}
assert report['deterministic']['passed']==1193 and report['service_integration']['passed']==60
assert report['phase_a_regression']['passed']==378 and report['phase_b_reranker_focused']['passed']==48
assert report['new_phase_c_deterministic']['passed']==41
write_json(out/'test-summary.json',report)
for name in ('normal-post-scale.log','normal-post-scale.xml','service-post-scale.log','service-post-scale.xml','service-native-retry.log','service-native-retry.xml'):
 shutil.copyfile(logs/name,out/name)
shutil.copytree(fixture,out/'reranker-native-fixture',ignore=shutil.ignore_patterns('*-shm','*-wal'))
shutil.copyfile(__file__,out/'record-verification.py')
# Earlier TDD logs remain in the preserved pre-refactor verification directory.
shutil.copytree(logs/'freshness-post-scale',out/'freshness-post-scale',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
write_json(out/'commands.json',{
 'normal':".venv/bin/python -m pytest -m 'not integration' -q --junitxml=evaluation/results/phase-c-v1/normal-post-scale.xml",
 'normal_environment':{'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'},
 'integration':'.venv/bin/python -m pytest -m integration -q --junitxml=evaluation/results/phase-c-v1/service-post-scale.xml',
 'integration_environment':{'ARKB_RUN_MODEL_TESTS':'1','ARKB_RUN_RERANKER_TESTS':'1','ARKB_QDRANT_URL':'http://127.0.0.1:6340',
 'ARKB_RERANKER_OFFLINE':'1','ARKB_RERANKER_CACHE':str(root/'.arkb/models'),
 'ARKB_RERANKER_TEST_DB':str(root/'evaluation/results/p0-current-20260910-r2/index.sqlite'),
 'ARKB_RERANKER_TEST_VAULT':'p0-current-20260910-r2','ARKB_RERANKER_TEST_QDRANT_URL':'http://127.0.0.1:32776',
 'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'},
 'retry':".venv/bin/python -m pytest 'tests/retrieval/integration/test_qwen_reranker_model.py::test_runtime_reranks_real_snapshot_preserving_evidence_and_filters[semantic]' -q --junitxml=evaluation/results/phase-c-v1/service-native-retry.xml",
 'retry_environment':'Same as integration_environment, except ARKB_RERANKER_TEST_DB='+str(fixture/'index.sqlite')+' and ARKB_RERANKER_TEST_QDRANT_URL=http://127.0.0.1:6340',
 'freshness':'.venv/bin/python evaluation/experiments/run_p4_freshness_live.py --output evaluation/results/phase-c-v1/freshness-post-scale --qdrant-url http://127.0.0.1:6340',
 'note':'Use fresh output paths for future runs. Counts are from JUnit cases; subsets overlap.'})
write_json(out/'source-files.json',{p.relative_to(root).as_posix():digest(p) for p in (root/'src').rglob('*.py')})
for p in (out/'freshness-post-scale/measured-source/src').rglob('*.py'):
 assert digest(p)==digest(root/'src'/p.relative_to(out/'freshness-post-scale/measured-source/src'))
write_json(out/'checksums.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p!=out/'checksums.json'})
verify_checksums(out)
print({k:v for k,v in report.items() if isinstance(v,dict) and 'cases' in v})

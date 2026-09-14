"""Freeze independent selections, label-isolated inputs and per-context indexes."""
import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from arkb.evaluation.external import digest, write_json, opaque_source
from .selection import build_selection, schedule
from .contract import ARMS, BASE_INSTRUCTION, BUDGET, MODEL, OPTIONS, THINK, rendered_prompt

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'evaluation/agentic-tools/v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Refusing to overwrite frozen input: ' + str(path))
    else:
        write_json(path, value)


def prepare_inputs(output, storage):
    output.mkdir(parents=True, exist_ok=True)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for name in ('inference', 'scoring', 'contexts', 'context-indexes', 'provenance'):
        (output / name).mkdir(exist_ok=True)
    populations, inputs, cases = {}, {}, {}
    for ds in ('browsecomp-plus', 'fiqa', 'nfcorpus'):
        preserved = ROOT / 'evaluation/phase-c/v1/validation' / (ds + '-preserved-index.json')
        index = json.loads(preserved.read_text())
        data = Path(index['dataset_directory'])
        manifest = json.loads((data / 'manifest.json').read_text())
        if digest(data / 'queries.json') != manifest['files']['queries.json']:
            raise ValueError('Query data changed.')
        queries = json.loads((data / 'queries.json').read_text())
        if set(queries) != set(manifest['query_ids']):
            raise ValueError('Query population changed.')
        populations[ds] = manifest['query_ids']
        cases[ds] = queries
        inputs[ds] = {'dataset': ds, 'data': str(data), 'corpus': index['live_corpus'],
                      'sqlite': index['sqlite'], 'vault_id': index['index_manifest']['vault_id'],
                      'index_manifest': index['index_manifest'], 'backend': index['backend'],
                      'preserved_manifest_sha256': digest(preserved),
                      'data_manifest_sha256': digest(data / 'manifest.json'),
                      'sqlite_sha256': index['sqlite_sha256'],
                      'snapshot_sha256': index['snapshot_sha256']}
    musique = ROOT / 'evaluation/results/p4-musique-agent-v3/selected.json'
    selected = json.loads(musique.read_text())
    pairs = defaultdict(list)
    for i, row in enumerate(selected['rows']):
        pairs[row['id']].append((i, row))
    if any(len(rs) != 2 or {r['answerable'] for _, r in rs} != {True, False} for rs in pairs.values()):
        raise ValueError('Incomplete MuSiQue input pairs.')
    selection = build_selection(populations, pairs)
    json_write_once(PUBLIC / 'selection.json', selection)
    json_write_once(output / 'selection.json', selection)
    design = {'schema': 'agentic-tools-v1-design', 'status': 'pilot_preparation',
              'arms': [asdict(a) for a in ARMS], 'model': MODEL, 'options': OPTIONS, 'think': THINK,
              'budget': asdict(BUDGET), 'turns': 8, 'harness_deadline_seconds': 360,
              'rerank': False, 'candidate_k': 20, 'fixed_top_k': 20,
              'base_instruction': BASE_INSTRUCTION,
              'prompts': {a.id: rendered_prompt(a) for a in ARMS},
              'selection_sha256': digest(output / 'selection.json'),
              'pilot_attempts': 98, 'core_attempts': 5460,
              'note': 'Design freeze only; final executable protocol requires runtime, judge and pilot evidence.'}
    json_write_once(PUBLIC / 'design.json', design)
    json_write_once(output / 'design.json', design)
    scenarios, gold = {}, {}
    for split in ('pilot', 'core'):
        for r in schedule(selection, split):
            identity = (r['dataset'], r['id'], r['variant'])
            scenario_id = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            if scenario_id in scenarios:
                continue
            ds, qid, variant = identity
            scenario = {'scenario_id': scenario_id, 'dataset': ds, 'id': qid, 'variant': variant, 'split': split}
            if ds != 'musique':
                scenario['query'] = cases[ds][qid]
            else:
                original_i, row = pairs[qid][int(variant[1:])]
                old_context = ROOT / 'evaluation/results/p4-musique-agent-v3/contexts' / str(original_i)
                directory = output / 'contexts' / scenario_id
                directory.mkdir(exist_ok=True)
                contents, source_map = {}, {}
                for paragraph in row['paragraphs']:
                    source = opaque_source(f'musique-{original_i}', str(paragraph['idx']))
                    contents[source] = '# ' + ' '.join(paragraph['title'].splitlines()).strip() + '\n' + paragraph['paragraph_text'].strip()
                    if (old_context / source).read_text() != contents[source]:
                        raise ValueError('Original MuSiQue context changed.')
                    dest = directory / source
                    if dest.exists() and dest.read_text() != contents[source]:
                        raise ValueError('New MuSiQue context drift.')
                    if not dest.exists():
                        dest.write_text(contents[source])
                    source_map[source] = paragraph['idx']
                if {p.name for p in directory.iterdir()} != set(contents):
                    raise ValueError('Unexpected context files.')
                scenario.update(query=row['question'], corpus=str(directory),
                                context_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in contents.items()},
                                sqlite=str(output / 'context-indexes' / (scenario_id + '.sqlite')),
                                vault_id='agentic-musique')
                gold[scenario_id] = {'id': qid, 'answer': row['answer'], 'answer_aliases': row['answer_aliases'],
                                    'answerable': row['answerable'], 'source_map': source_map,
                                    'support_idxs': [p['idx'] for p in row['paragraphs'] if p['is_supporting']]}
            scenarios[scenario_id] = scenario
    assert len(scenarios) == 274
    # Explicit projection: neither support labels, answers nor answerability can
    # enter the model-facing scenario packet or materialized source directories.
    json_write_once(output / 'inference/scenarios.json', scenarios)
    json_write_once(output / 'scoring/musique.json', gold)
    inputs['musique'] = {'source_selection_sha256': digest(musique), 'selected_contexts': len(gold),
                         'raw_source': str(musique), 'scope': 'original supplied context, whole paragraphs'}
    json_write_once(output / 'inputs.json', inputs)
    json_write_once(PUBLIC / 'inputs.json', inputs)
    json_write_once(output / 'pilot-schedule.json', schedule(selection, 'pilot'))
    json_write_once(output / 'core-schedule.json', schedule(selection, 'core'))
    # Read source IDs by streaming the corpus. Do not retain corpus text or
    # expose relevance labels through any inference-facing metadata.
    for ds in populations:
        dest = output / 'scoring' / (ds + '.json')
        if dest.exists():
            continue
        data = Path(inputs[ds]['data'])
        source_map = {}
        with (data / 'corpus.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                source_map[opaque_source(ds, row['id'])] = row['id']
        labels = {'qrels': json.loads((data / 'qrels.json').read_text()), 'source_map': source_map}
        if ds == 'browsecomp-plus':
            labels['gold_qrels'] = json.loads((data / 'gold-qrels.json').read_text())
        json_write_once(dest, labels)
    write_json(output / 'preparation-status.json', {'status': 'inputs_prepared', 'updated_at': utc(),
                                                  'scenarios': len(scenarios), 'musique_contexts': len(gold),
                                                  'pilot_attempts': 98, 'core_attempts': 5460})
    print(json.dumps({'stage': 'inputs_prepared', 'scenarios': len(scenarios), 'musique_contexts': len(gold)}), flush=True)


def prepare_context_indexes(output):
    from arkb.runtime import Runtime
    from arkb.config import RuntimeConfig
    from arkb.knowledge.sqlite import SQLiteStorage
    from time import perf_counter
    scenarios = json.loads((output / 'inference/scenarios.json').read_text())
    path = output / 'context-indexes.json'
    reports = json.loads(path.read_text()) if path.exists() else {}
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(),
                               qdrant_url='http://127.0.0.1:6340')) as runtime:
        for sid, case in scenarios.items():
            if case['dataset'] != 'musique':
                continue
            for source, expected in case['context_sha256'].items():
                if digest(Path(case['corpus']) / source) != expected:
                    raise ValueError('MuSiQue context identity changed.')
            if sid in reports:
                with SQLiteStorage(Path(case['sqlite']), read_only=True) as storage:
                    assert asdict(storage.active_manifest(case['vault_id'])) == reports[sid]['build']['manifest']
                continue
            if Path(case['sqlite']).exists():
                raise ValueError('Unrecorded context indexing attempt; preserve and diagnose before resuming.')
            start = perf_counter()
            build = runtime.index(db=Path(case['sqlite']), vault_id=case['vault_id'],
                                  notes_dir=Path(case['corpus']), chunking='none', batch_size=32)
            reports[sid] = {'build': asdict(build), 'elapsed_ms': (perf_counter() - start) * 1000,
                            'sqlite_sha256': digest(case['sqlite'])}
            write_json(path, reports)
            print(json.dumps({'stage': 'context_index_ready', 'completed': len(reports), 'total': 66}), flush=True)
    assert len(reports) == 66
    write_json(output / 'preparation-status.json', {'status': 'contexts_ready', 'updated_at': utc(), 'contexts': 66})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--storage', type=Path, default=Path('/Volumes/ARKBPhaseC'))
    p.add_argument('--contexts', action='store_true')
    a = p.parse_args()
    if a.contexts:
        prepare_context_indexes(a.output.resolve())
    else:
        prepare_inputs(a.output.resolve(), a.storage.resolve())


if __name__ == '__main__':
    main()

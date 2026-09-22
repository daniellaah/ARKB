"""Build the development set: scenarios, labels and scopes, from IDs, hashes and corpus text only.

The core slices (v2, exact-v2) need nothing beyond the repository. The optional
slices take their inputs from the frozen pilot-v3 selection on the external
volume and are built only when it is mounted (`--core-only` skips them).
No answer, judgment or score is consulted when choosing scenarios.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from arkb.knowledge.documents import load_notes

from ..common import ROOT, digest, digest_text, write_json

IDENTITY = 'arkb-devset-v2'
# Selection hashes keep their original identities so the chosen terms and queries stay those of devset-v1.
TERM_IDENTITY = 'arkb-devloop-v1'
SELECTION_IDENTITY = 'agentic-tools-v1|20260912'
DEVSET = ROOT / 'evaluation/devset'
V2 = DEVSET / 'v2-pilot'
TRANSLATIONS = DEVSET / 'v2-pilot-english-queries.json'
V2_INDEX = Path('.arkb/devset/v2-pilot/index.sqlite')
V2_VAULT = 'devset-v2-pilot'
PILOT_V3 = Path('/Volumes/ARKBPhaseC/agentic-tools-v1/pilot-v3')
QDRANT_URL = 'http://127.0.0.1:6340'
EXACT_STRATA = {'df_2_5': (2, 5, 8), 'df_6_12': (6, 12, 8), 'df_13_30': (13, 30, 4)}
PHRASE_STRATUM = (2, 10, 4)
RECALL_PER_DATASET = 20
LONG_DOCUMENT_QUERIES = 10
STOPWORDS = set('''about above after again against almost already although always among another anyone around because
before being below between beyond cannot certain could during either every everything further having however
indeed instead itself little might nothing often others otherwise perhaps rather really should since something
still their there these things those though through toward under unless until where whether which while whole
whose within without would'''.split())
EXACT_QUERY = ("List every note whose body contains the exact text '{term}' as a case-sensitive substring. "
               "Return the filename of every matching note. If nothing matches, say so explicitly.")


def term_key(dataset, term):
    return digest_text(f'{TERM_IDENTITY}|{dataset}|{term}')


def selection_key(dataset, stratum, identifier):
    return digest_text(f'{SELECTION_IDENTITY}|{dataset}|{stratum}|{identifier}')


def exact_tasks(corpus_dir, dataset, *, scope=None, optional=False, strata=EXACT_STRATA, phrase=PHRASE_STRATUM):
    """Literal enumeration tasks whose truth is the substring truth `match` implements."""
    notes = load_notes(Path(corpus_dir))
    bodies = {note.source: note.content for note in notes}
    word = re.compile(r'[A-Za-z][a-z]{5,}')
    token_df, bigram_df = Counter(), Counter()
    for body in bodies.values():
        tokens = word.findall(body)
        token_df.update(set(tokens))
        bigram_df.update({f'{a} {b}' for a, b in zip(tokens, tokens[1:]) if a.lower() not in STOPWORDS and b.lower() not in STOPWORDS})

    def substring_truth(term):
        sources = sorted(source for source, body in bodies.items() if term in body)
        return sources, sum(body.count(term) for body in bodies.values())

    tasks = []
    def fill(candidates, low, high, count, stratum):
        chosen = 0
        for term in sorted(candidates, key=lambda t: term_key(dataset, t)):
            if chosen == count:
                break
            sources, occurrences = substring_truth(term)
            if not low <= len(sources) <= high:
                continue
            tasks.append({'id': f'exact-{dataset}-{term_key(dataset, term)[:12]}', 'slice': f'exact-{dataset}', 'scope': scope or dataset,
                          'task_type': 'exact_enumeration', 'stratum': stratum, 'term': term,
                          'query': EXACT_QUERY.format(term=term), **({'optional': True} if optional else {}),
                          'labels': {'expected_sources': sources, 'occurrences': occurrences,
                                     'distinct_sources': len(sources)}})
            chosen += 1
        if chosen != count:
            raise ValueError(f'Insufficient candidates for stratum {stratum}.')
    for stratum, (low, high, count) in strata.items():
        fill([t for t, df in token_df.items() if low <= df <= high and t.lower() not in STOPWORDS], low, high, count, stratum)
    low, high, count = phrase
    fill([p for p, df in bigram_df.items() if low <= df <= high], low, high, count, 'phrase_2_10')
    return tasks


def v2_tasks():
    translations = json.loads(TRANSLATIONS.read_text())['queries']
    tasks = []
    for line in (V2 / 'queries.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['id'] not in translations:
            raise ValueError('Missing English translation: ' + row['id'])
        tasks.append({'id': row['id'], 'slice': 'v2', 'scope': 'v2-pilot', 'task_type': row['task_type'],
                      'answerability': row['answerability'], 'intent_family_id': row['intent_family_id'],
                      'query': translations[row['id']], 'query_zh': row['query'],
                      'labels': {'dataset': 'evaluation/devset/v2-pilot', 'case_id': row['id']}})
    return tasks


def recall_tasks(dataset, inputs, selection, scoring):
    stratum = selection['tracks'][dataset][0]
    ordered = sorted(stratum['pilot'] + stratum['reserves'], key=lambda x: (selection_key(dataset, 'all', x), x))
    chosen = ordered[:RECALL_PER_DATASET]
    queries = json.loads((Path(inputs[dataset]['data']) / 'queries.json').read_text())
    tasks = []
    for identifier in chosen:
        tasks.append({'id': f'recall-{dataset}-{identifier}', 'slice': f'recall-{dataset}', 'scope': dataset, 'optional': True,
                      'task_type': 'evidence_recall', 'query': queries[identifier], 'source_id': identifier,
                      'exposure': 'pilot' if identifier in stratum['pilot'] else 'reserve',
                      'labels': {'qrels': scoring[dataset]['qrels'][identifier], 'source_map': f'scoring/{dataset}.json'}})
    return tasks


def long_document_tasks(scenarios, selection, scoring, inputs):
    """Long-document slice: BrowseComp pilot queries plus the first reserves in hash order.

    Preparing the 100,195-document exact cache takes about ten minutes; the slice
    tests read expansion and evidence packing on long web pages.
    """
    stratum = selection['tracks']['browsecomp-plus'][0]
    ordered = sorted(stratum['pilot'] + stratum['reserves'], key=lambda x: (selection_key('browsecomp-plus', 'all', x), x))
    chosen = ordered[:LONG_DOCUMENT_QUERIES]
    by_identity = {(v['dataset'], v['id'], v['variant']): v for v in scenarios.values()}
    queries = json.loads((Path(inputs['browsecomp-plus']['data']) / 'queries.json').read_text())
    tasks = []
    for identifier in chosen:
        case = by_identity.get(('browsecomp-plus', identifier, 'v0'))
        tasks.append({'id': f'long-browsecomp-{identifier}', 'slice': 'long-browsecomp', 'scope': 'browsecomp-plus', 'optional': True,
                      'task_type': 'long_document_answer', 'query': case['query'] if case else queries[identifier], 'source_id': identifier,
                      'exposure': 'pilot' if identifier in stratum['pilot'] else 'reserve',
                      'labels': {'qrels': scoring['browsecomp-plus']['qrels'][identifier],
                                 'gold_qrels': scoring['browsecomp-plus']['gold_qrels'][identifier],
                                 'source_map': 'scoring/browsecomp-plus.json'}})
    return tasks


def musique_tasks(scenarios, selection, gold):
    by_identity = {(v['dataset'], v['id'], v['variant']): v for v in scenarios.values()}
    pairs = [case['id'] for case in scenarios.values() if case['dataset'] == 'musique' and case['split'] == 'pilot' and case['variant'] == 'v0']
    for stratum, count in (('2', 3), ('3', 2), ('4', 2)):
        entry = next(s for s in selection['tracks']['musique'] if s['stratum'] == stratum)
        pairs.extend(entry['core'][:count])
    tasks = []
    for pair in pairs:
        for variant in ('v0', 'v1'):
            case = by_identity[('musique', pair, variant)]
            tasks.append({'id': f'musique-{case["scenario_id"][:12]}-{variant}', 'slice': 'musique', 'scope': case['scenario_id'],
                          'optional': True, 'task_type': 'multi_hop', 'query': case['query'], 'pair_id': pair, 'variant': variant,
                          'exposure': case['split'], 'corpus': case['corpus'], 'sqlite': case['sqlite'],
                          'vault_id': case['vault_id'], 'context_sha256': case['context_sha256'],
                          'labels': {'gold': gold[case['scenario_id']]}})
    return tasks


def carry_over(destination):
    """Optional scenarios, labels and scopes of an existing devset, unchanged."""
    if not (destination / 'scenarios.json').exists():
        return None
    scenarios = json.loads((destination / 'scenarios.json').read_text())
    labels = json.loads((destination / 'labels.json').read_text())
    scopes = json.loads((destination / 'scopes.json').read_text())
    manifest = json.loads((destination / 'manifest.json').read_text())
    tasks = [{**s, 'optional': True, 'labels': labels[s['id']]} for s in scenarios
             if s['slice'] not in ('v2', 'exact-v2')]
    kept = {s['scope'] for s in tasks}
    return {'tasks': tasks, 'scopes': {k: v for k, v in scopes.items() if k in kept},
            'sources': {k: v for k, v in manifest['sources'].items() if k.startswith('pilot_v3')}
                       | {'optional_carried_from': manifest['identity']}}


def core_scopes():
    return {'v2-pilot': {'corpus': str((V2 / 'corpus').relative_to(ROOT)), 'sqlite': str(V2_INDEX), 'vault_id': V2_VAULT, 'prepare_exact': True}}


def build(destination, *, core_only=False):
    tasks = v2_tasks() + exact_tasks(V2 / 'corpus', 'v2', scope='v2-pilot')
    scopes = core_scopes()
    sources = {'v2_queries': digest(V2 / 'queries.jsonl'), 'v2_evidence': digest(V2 / 'evidence.jsonl'),
               'v2_qrels': digest(V2 / 'qrels.jsonl'), 'translations': digest(TRANSLATIONS)}
    scoring, carried = {}, None
    if not core_only and not PILOT_V3.exists():
        # Without the volume the optional slices cannot be rebuilt; keep the committed ones as they are.
        carried = carry_over(destination)
        if carried is None:
            raise FileNotFoundError(f'{PILOT_V3} is not mounted and {destination} holds no devset; pass --core-only.')
        tasks += carried['tasks']
        scopes.update(carried['scopes'])
        sources.update(carried['sources'])
    elif not core_only:
        inputs = json.loads((PILOT_V3 / 'inputs.json').read_text())
        selection = json.loads((PILOT_V3 / 'selection.json').read_text())
        scenarios = json.loads((PILOT_V3 / 'inference/scenarios.json').read_text())
        scoring = {ds: json.loads((PILOT_V3 / 'scoring' / f'{ds}.json').read_text()) for ds in ('nfcorpus', 'fiqa', 'browsecomp-plus')}
        musique_gold = json.loads((PILOT_V3 / 'scoring/musique.json').read_text())
        tasks += exact_tasks(inputs['nfcorpus']['corpus'], 'nfcorpus', optional=True)
        for ds in ('nfcorpus', 'fiqa'):
            tasks += recall_tasks(ds, inputs, selection, scoring)
        tasks += musique_tasks(scenarios, selection, musique_gold)
        tasks += long_document_tasks(scenarios, selection, scoring, inputs)
        for ds in ('nfcorpus', 'fiqa', 'browsecomp-plus'):
            scopes[ds] = {'corpus': inputs[ds]['corpus'], 'sqlite': inputs[ds]['sqlite'], 'vault_id': inputs[ds]['vault_id'],
                          'index_manifest': inputs[ds]['index_manifest'], 'prepare_exact': True}
        sources.update(pilot_v3_selection=digest(PILOT_V3 / 'selection.json'), pilot_v3_scenarios=digest(PILOT_V3 / 'inference/scenarios.json'),
                       pilot_v3_inputs=digest(PILOT_V3 / 'inputs.json'))
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate development task IDs.')
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'scenarios.json', [{k: v for k, v in t.items() if k != 'labels'} for t in tasks])
    write_json(destination / 'labels.json', {t['id']: t['labels'] for t in tasks})
    write_json(destination / 'scopes.json', scopes)
    for ds, value in scoring.items():
        write_json(destination / 'scoring' / f'{ds}.json', {'source_map': value['source_map']})
    counts = Counter(t['slice'] for t in tasks)
    write_json(destination / 'manifest.json', {
        'schema': 'arkb-devset-v2', 'identity': IDENTITY, 'tasks': len(tasks), 'by_slice': dict(counts),
        'core_slices': sorted({t['slice'] for t in tasks if not t.get('optional')}),
        'optional_slices': sorted({t['slice'] for t in tasks if t.get('optional')}),
        'sources': sources, 'files': {name: digest(destination / name) for name in ('scenarios.json', 'labels.json', 'scopes.json')},
        'exposure': 'development material: every ID here is exposed; never a holdout',
        'selection': 'IDs and corpus text statistics only; no outcomes, answers or scores were consulted'})
    return counts


def build_v2_index(force=False):
    from arkb.config import RuntimeConfig
    from arkb.runtime import Runtime
    from arkb.knowledge.models import QdrantConfig
    index = ROOT / V2_INDEX
    index.parent.mkdir(parents=True, exist_ok=True)
    with Runtime(RuntimeConfig(offline=True, tokenizer_cache=(ROOT / '.uv-cache/tokenizers').resolve(), qdrant_url=QDRANT_URL)) as runtime:
        report = runtime.index(db=index, vault_id=V2_VAULT, notes_dir=V2 / 'corpus',
                               qdrant_config=QdrantConfig(url=QDRANT_URL), force=force)
        return {'index_version': report.manifest.index_version, 'documents': report.manifest.document_count,
                'chunks': report.manifest.chunk_count, 'reused': report.reused_index}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=DEVSET)
    parser.add_argument('--index', action='store_true', help='build or verify the v2 pilot corpus index')
    parser.add_argument('--core-only', action='store_true', help='skip the optional slices that need the external volume')
    args = parser.parse_args()
    if args.index:
        print(json.dumps({'v2_index': build_v2_index()}))
    counts = build(args.destination.resolve(), core_only=args.core_only)
    print(json.dumps({'built': str(args.destination), 'by_slice': dict(counts)}))


if __name__ == '__main__':
    main()

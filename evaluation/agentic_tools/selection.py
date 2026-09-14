"""Outcome-independent selection and balanced scheduling, using IDs only."""
from collections import defaultdict
import hashlib
import re

from .contract import ARMS

SEED = 20260912
IDENTITY = 'agentic-tools-v1'


def key(dataset, stratum, identifier):
    return hashlib.sha256(f'{IDENTITY}|{SEED}|{dataset}|{stratum}|{identifier}'.encode('utf-8')).hexdigest()


def select(ids, dataset, stratum, pilot, core):
    ids = list(ids)
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('Selection requires unique nonblank string IDs.')
    if len(ids) < pilot + core:
        raise ValueError('Insufficient registered population.')
    ordered = sorted(ids, key=lambda x: (key(dataset, stratum, x), x))
    return {'stratum': str(stratum), 'population_count': len(ids),
            'pilot': ordered[:pilot], 'core': ordered[pilot:pilot + core],
            'reserves': ordered[pilot + core:]}


def build_selection(populations, musique_pairs):
    if set(populations) != {'browsecomp-plus', 'fiqa', 'nfcorpus'}:
        raise ValueError('All three full-corpus tracks are required.')
    tracks = {}
    for dataset, counts in [('browsecomp-plus', (4, 100)), ('fiqa', (2, 50)), ('nfcorpus', (2, 50))]:
        tracks[dataset] = [select(populations[dataset], dataset, 'all', *counts)]
    by_hop = defaultdict(list)
    for identifier in musique_pairs:
        match = re.match(r'^([234])hop(?:[123])?__', identifier)
        if match is None:
            raise ValueError('Unknown MuSiQue hop stratum.')
        hop = match.group(1)
        by_hop[hop].append(identifier)
    if {h: len(xs) for h, xs in by_hop.items()} != {'2': 34, '3': 33, '4': 33}:
        raise ValueError('MuSiQue population differs from the registered 100 pairs.')
    tracks['musique'] = [select(by_hop[h], 'musique', h, 1, 10) for h in ('2', '3', '4')]
    return {'schema': IDENTITY + '-selection-v1', 'seed': SEED,
            'algorithm': 'sha256(UTF8(identity|seed|dataset|stratum|id)), then ID', 'tracks': tracks}


def schedule(selection, split):
    if split not in ('pilot', 'core'):
        raise ValueError('Unknown split.')
    repetitions = 1 if split == 'pilot' else 3
    schedule_rows, unit_index = [], 0
    for dataset, strata in selection['tracks'].items():
        for stratum in strata:
            for identifier in stratum[split]:
                for repetition in range(repetitions):
                    offset = (unit_index + repetition) % len(ARMS)
                    rotated = ARMS[offset:] + ARMS[:offset]
                    for arm in rotated:
                        for variant in (('v0', 'v1') if dataset == 'musique' else ('v0',)):
                            schedule_rows.append({'dataset': dataset, 'id': identifier, 'stratum': stratum['stratum'],
                                                  'variant': variant, 'arm': arm.id, 'repetition': repetition,
                                                  'unit_index': unit_index})
                unit_index += 1
    return schedule_rows


def attempt_key(protocol_hash, row):
    import json
    fields = ['dataset', 'id', 'variant', 'arm', 'repetition']
    return hashlib.sha256(json.dumps([protocol_hash, *[row[k] for k in fields]],
                                    ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()

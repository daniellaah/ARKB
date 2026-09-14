"""Registered query/pair bootstrap; repetitions are not independent questions."""
from collections import defaultdict

import numpy as np

from .contract import ARMS
from .selection import SEED


def paired_comparisons(units, *, primary=False, resamples=20000):
    """units: [{id, stratum, values: {arm: mean over repetitions}}]."""
    if not units:
        raise ValueError('No independent units.')
    units = sorted(units, key=lambda u: (u['stratum'], u['id']))
    if len({u['id'] for u in units}) != len(units):
        raise ValueError('Repeated trials cannot be bootstrap units.')
    arm_ids = [a.id for a in ARMS]
    if any(set(u['values']) != set(arm_ids) for u in units):
        raise ValueError('A registered arm is missing.')
    if any(v is None for u in units for v in u['values'].values()):
        return {'status': 'pending_scores', 'independent_units': len(units), 'contrasts': [],
                'reason': 'Unresolved scoring is not converted to a model failure or dropped.'}
    values = np.array([[u['values'][a] for a in arm_ids] for u in units], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Undefined metric entered bootstrap.')
    strata = defaultdict(list)
    for i, u in enumerate(units):
        strata[u['stratum']].append(i)
    rng = np.random.default_rng(SEED)
    indices = np.concatenate([rng.choice(ix, size=(resamples, len(ix)), replace=True)
                              for _, ix in sorted(strata.items())], axis=1)
    means = values[indices].mean(axis=1)
    contrasts = []
    for reference in ['A-M', 'A-B', 'A-S', 'A-H', 'F-S', 'F-H']:
        left, right = arm_ids.index('A-All'), arm_ids.index(reference)
        delta = values[:, left] - values[:, right]
        boot = means[:, left] - means[:, right]
        nominal = np.quantile(boot, [.025, .975]).tolist()
        adjusted = np.quantile(boot, [.00625, .99375]).tolist()
        is_primary = primary and reference.startswith('A-')
        contrasts.append({'comparison': 'A-All minus ' + reference, 'mean_difference': float(delta.mean()),
                          'nominal_95_ci': nominal, 'adjusted_98_75_ci': adjusted if is_primary else None,
                          'wins': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()),
                          'losses': int((delta < -1e-12).sum()),
                          'primary_family': is_primary,
                          'meaningful_gain_demonstrated': bool(delta.mean() >= .05 and adjusted[0] > 0) if is_primary else None})
    return {'status': 'complete', 'independent_units': len(units), 'resamples': resamples, 'seed': SEED,
            'stratum_sizes': {k: len(v) for k, v in strata.items()},
            'arm_means': {a: float(values[:, i].mean()) for i, a in enumerate(arm_ids)},
            'contrasts': contrasts,
            'advantage_over_every_restricted_arm': all(c['meaningful_gain_demonstrated'] for c in contrasts[:4]) if primary else None,
            'coverage_note': 'Finite-sample bootstrap coverage is approximate; no pooled cross-dataset score.'}


def query_units(rows, metric, *, variants=('v0',)):
    grouped = defaultdict(list)
    strata = {}
    for r in rows:
        s = r['schedule']
        grouped[(s['id'], s['arm'])].append(r)
        strata[s['id']] = s['stratum']
    result = []
    for identifier in sorted(strata):
        values = {}
        for arm in ARMS:
            cases = grouped[(identifier, arm.id)]
            identities = [(r['schedule']['variant'], r['schedule']['repetition']) for r in cases]
            if len(set(identities)) != len(identities) or set(identities) != {(v, rep) for v in variants for rep in range(3)}:
                raise ValueError('A query is missing a repetition or contains duplicates.')
            if not variants:
                raise ValueError('An arm has no repeated observations.')
            items = [r[metric] for r in cases]
            values[arm.id] = None if any(v is None for v in items) else float(np.mean(items))
        result.append({'id': identifier, 'stratum': strata[identifier], 'values': values})
    return result

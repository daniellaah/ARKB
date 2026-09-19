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


def query_units(rows, metric, *, variants=('v0',), repetitions=(0, 1, 2)):
    """One unit per question; the registered repetitions are averaged, never counted as questions."""
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
            if len(set(identities)) != len(identities) or set(identities) != {(v, rep) for v in variants for rep in repetitions}:
                raise ValueError('A query is missing a repetition or contains duplicates.')
            if not variants:
                raise ValueError('An arm has no repeated observations.')
            items = [r[metric] for r in cases]
            values[arm.id] = None if any(v is None for v in items) else float(np.mean(items))
        result.append({'id': identifier, 'stratum': strata[identifier], 'values': values})
    return result


def session_variability(rows, metric, *, repetitions=(0, 1, 2)):
    """One-way random-effects decomposition on a registered repeat subset.

    Returns per-arm and per-primary-contrast between-question and within-question
    variance components and their ratio. This is descriptive evidence about
    session variability on a small registered subset; it is not a population
    intraclass correlation and never enters the primary contrasts.
    """
    if len(repetitions) < 2:
        raise ValueError('Session variability needs at least two registered repetitions.')
    grouped = defaultdict(dict)
    for r in rows:
        s = r['schedule']
        if s['repetition'] in grouped[(s['id'], s['arm'])]:
            raise ValueError('Duplicate repetition in the repeat subset.')
        grouped[(s['id'], s['arm'])][s['repetition']] = r[metric]
    ids = sorted({identifier for identifier, _ in grouped})
    arm_ids = [a.id for a in ARMS]
    if len(ids) < 2:
        raise ValueError('Session variability needs at least two questions.')
    for identifier in ids:
        for arm in arm_ids:
            if set(grouped.get((identifier, arm), {})) != set(repetitions):
                raise ValueError('Repeat subset is incomplete.')
    if any(v is None for cell in grouped.values() for v in cell.values()):
        return {'status': 'pending_scores', 'questions': len(ids), 'repetitions': list(repetitions)}
    k = len(repetitions)
    matrix = {arm: np.array([[grouped[(identifier, arm)][rep] for rep in repetitions] for identifier in ids], dtype=float)
              for arm in arm_ids}

    def decompose(values):
        n = values.shape[0]
        question_means = values.mean(axis=1)
        grand = values.mean()
        ms_between = k * ((question_means - grand) ** 2).sum() / (n - 1)
        ms_within = ((values - question_means[:, None]) ** 2).sum() / (n * (k - 1))
        between = max(0.0, (ms_between - ms_within) / k)
        total = between + ms_within
        return {'grand_mean': float(grand), 'between_question_variance': float(between),
                'within_question_variance': float(ms_within),
                'repeat_correlation': float(between / total) if total > 0 else None,
                'questions_with_identical_repetitions': int((values.min(axis=1) == values.max(axis=1)).sum())}
    return {'status': 'complete', 'questions': len(ids), 'repetitions': list(repetitions),
            'arms': {arm: decompose(matrix[arm]) for arm in arm_ids},
            'primary_contrasts': {'A-All minus ' + ref: decompose(matrix['A-All'] - matrix[ref])
                                  for ref in ('A-M', 'A-B', 'A-S', 'A-H')},
            'scope': 'Descriptive one-way random-effects estimates on a small registered subset; not a population intraclass correlation; no best-of-k; excluded from primary contrasts.'}

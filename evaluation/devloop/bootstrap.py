"""Paired bootstrap over development-loop runs of any arms; nominal intervals only.

The registered `statistics.paired_comparisons` is wired to the seven registered
arms. This one takes any runs, pairs them per scenario, and reports per-slice
mean differences with slice-stratified percentile intervals. Development
material, single runs, no multiplicity correction: engineering evidence only.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

SEED = 20260912
RESAMPLES = 20000
# Metrics per slice; dotted paths read nested score fields.
SLICE_METRICS = {
    'v2': ['evidence_coverage_delivered', 'source_recall_cited', 'behavior.answered',
           'behavior.no_retrieval_respected', 'behavior.abstained'],
    'exact-nfcorpus': ['completeness_cited', 'complete_and_exact_cited', 'spurious_cited'],
    'recall-nfcorpus': ['positive_recall_delivered', 'positive_recall_returned', 'positive_recall_submitted'],
    'recall-fiqa': ['positive_recall_delivered', 'positive_recall_returned', 'positive_recall_submitted'],
    'long-browsecomp': ['positive_recall_delivered', 'positive_recall_returned', 'positive_recall_submitted'],
    'musique': ['support_f1', 'answerability_correct', 'answer_f1_first_line', 'answer_em_first_line',
                'answer_f1', 'answer_em'],
}
COST_METRICS = ['elapsed_s', 'costs.model_requests', 'tool_calls', 'prompt_tokens', 'eval_tokens']


def metric_value(scores, metric):
    """A float or None; None drops the scenario from that metric's pairs."""
    if metric == 'elapsed_s':
        return scores['costs']['elapsed_ms'] / 1000
    if metric == 'tool_calls':
        return float(len(scores['tool_names']))
    if metric in ('prompt_tokens', 'eval_tokens'):
        usage = scores['costs'].get('usage') or {}
        key = 'prompt_eval_count' if metric == 'prompt_tokens' else 'eval_count'
        value = (usage.get(key) or {}).get('known_total')
        return None if value is None else float(value)
    value = scores
    for part in metric.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
    return None if value is None else float(value)


def paired_bootstrap(units, contrasts, *, seed=SEED, resamples=RESAMPLES):
    """units: [{id, stratum, values: {arm: float}}], every arm present; contrasts: [(left, right)]."""
    if not units:
        raise ValueError('No paired units.')
    units = sorted(units, key=lambda u: (u['stratum'], u['id']))
    if len({u['id'] for u in units}) != len(units):
        raise ValueError('A scenario appears twice.')
    arms = sorted({a for u in units for a in u['values']})
    if any(set(u['values']) != set(arms) for u in units):
        raise ValueError('An arm is missing from a unit.')
    values = np.array([[u['values'][a] for a in arms] for u in units], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Undefined metric entered bootstrap.')
    strata = defaultdict(list)
    for i, u in enumerate(units):
        strata[u['stratum']].append(i)
    rng = np.random.default_rng(seed)
    indices = np.concatenate([rng.choice(ix, size=(resamples, len(ix)), replace=True)
                              for _, ix in sorted(strata.items())], axis=1)
    means = values[indices].mean(axis=1)
    result = {'n': len(units), 'resamples': resamples, 'seed': seed,
              'stratum_sizes': {k: len(v) for k, v in sorted(strata.items())},
              'arm_means': {a: float(values[:, i].mean()) for i, a in enumerate(arms)}, 'contrasts': []}
    for left, right in contrasts:
        if left not in arms or right not in arms:
            raise ValueError('Contrast names an arm without a run: ' + f'{left} minus {right}')
        delta = values[:, arms.index(left)] - values[:, arms.index(right)]
        boot = means[:, arms.index(left)] - means[:, arms.index(right)]
        result['contrasts'].append({'comparison': f'{left} minus {right}', 'mean_difference': float(delta.mean()),
                                    'nominal_95_ci': np.quantile(boot, [.025, .975]).tolist(),
                                    'wins': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()),
                                    'losses': int((delta < -1e-12).sum())})
    return result


def load_runs(runs):
    """runs: {arm: path}; returns {arm: {scenario_id: row}} and the run metadata."""
    rows, meta = {}, {}
    for arm, path in runs.items():
        path = Path(path)
        rows[arm] = {r['scenario']['id']: r for r in
                     (json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines() if line.strip())}
        meta[arm] = json.loads((path / 'run.json').read_text())
    return rows, meta


def units_for(rows, metric, *, slices=None):
    """Scenarios shared by every arm with the metric defined in every arm, stratified by slice."""
    shared = sorted(set.intersection(*(set(r) for r in rows.values())))
    units = []
    for sid in shared:
        scenario = next(iter(rows.values()))[sid]['scenario']
        if slices and scenario['slice'] not in slices:
            continue
        values = {arm: metric_value(rows[arm][sid]['scores'], metric) for arm in rows}
        if any(v is None for v in values.values()):
            continue
        units.append({'id': sid, 'stratum': scenario['slice'], 'values': values})
    return units


def analyze(runs, contrasts, *, seed=SEED, resamples=RESAMPLES):
    rows, meta = load_runs(runs)
    slices = sorted({r['scenario']['slice'] for r in next(iter(rows.values())).values()})
    blocks = []
    for name in slices:
        for metric in SLICE_METRICS.get(name, []) + COST_METRICS:
            units = units_for(rows, metric, slices=[name])
            if units:
                blocks.append({'slice': name, 'metric': metric, **paired_bootstrap(units, contrasts, seed=seed, resamples=resamples)})
    if len(slices) > 1:
        for metric in COST_METRICS:
            units = units_for(rows, metric)
            if units:
                blocks.append({'slice': 'all (stratified)', 'metric': metric,
                               **paired_bootstrap(units, contrasts, seed=seed, resamples=resamples)})
    return {'runs': {arm: {'label': m['label'], 'model': m['model'], 'think': m['think'], 'git_head': m['git_head'],
                           'dirty': m['dirty']} for arm, m in meta.items()},
            'contrasts': [f'{a} minus {b}' for a, b in contrasts], 'seed': seed, 'resamples': resamples,
            'method': 'per-scenario paired differences; slice-stratified percentile bootstrap; nominal 95% intervals; '
                      'no multiplicity correction', 'blocks': blocks}


def markdown(report):
    arms = list(report['runs'])
    lines = ['# Paired bootstrap over development-loop runs', '',
             'Development material, single run per arm, nominal 95% percentile intervals, no multiplicity correction. '
             f"Seed {report['seed']}, {report['resamples']} resamples, stratified by slice.", '']
    for arm, m in report['runs'].items():
        lines.append(f"- {arm}: `{m['label']}` ({m['model']}, think={m['think']}, git={m['git_head'][:10]}{'+dirty' if m['dirty'] else ''})")
    lines += ['', '| slice | metric | n | ' + ' | '.join(arms) + ' | contrast | diff | 95% CI | win/tie/loss |',
              '| --- | --- | ---: | ' + ' | '.join('---:' for _ in arms) + ' | --- | ---: | --- | --- |']
    for block in report['blocks']:
        means = ' | '.join(f"{block['arm_means'][a]:.3f}" for a in arms)
        for i, c in enumerate(block['contrasts']):
            head = f"| {block['slice']} | {block['metric']} | {block['n']} | {means} |" if i == 0 else '| | | | ' + ' | '.join('' for _ in arms) + ' |'
            lo, hi = c['nominal_95_ci']
            lines.append(f"{head} {c['comparison']} | {c['mean_difference']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                         f"{c['wins']}/{c['ties']}/{c['losses']} |")
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, help='ARM=path to a run directory; repeatable')
    p.add_argument('--contrast', action='append', required=True, help='LEFT:RIGHT; repeatable')
    p.add_argument('--output', type=Path, help='Markdown destination; a .json sibling is written beside it')
    p.add_argument('--resamples', type=int, default=RESAMPLES)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    runs = dict(item.split('=', 1) for item in args.run)
    contrasts = [tuple(item.split(':', 1)) for item in args.contrast]
    report = analyze(runs, contrasts, seed=args.seed, resamples=args.resamples)
    text = markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=1) + '\n')
    print(text)


if __name__ == '__main__':
    main()

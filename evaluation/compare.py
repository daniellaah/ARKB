"""Paired comparison of runs: per-scenario differences with slice-stratified bootstrap intervals.

Development material, one run per configuration, nominal intervals, no
multiplicity correction: engineering evidence, not a quality claim.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .common import read_json

SEED = 20260912
RESAMPLES = 20000
# Metrics per slice; dotted paths read nested score fields.
SLICE_METRICS = {
    'v2': ['evidence_coverage_delivered', 'source_recall_cited', 'behavior.answered',
           'behavior.no_retrieval_respected', 'behavior.read_only_respected', 'behavior.abstained'],
    'exact-v2': ['completeness_cited', 'complete_and_exact_cited', 'spurious_cited'],
    'exact-nfcorpus': ['completeness_cited', 'complete_and_exact_cited', 'spurious_cited'],
    'recall-nfcorpus': ['positive_recall_delivered'],
    'recall-fiqa': ['positive_recall_delivered'],
    'long-browsecomp': ['positive_recall_delivered'],
    'musique': ['answerability_correct', 'answer_f1_first_line', 'answer_em_first_line', 'support_f1'],
}
COST_METRICS = ['elapsed_s', 'costs.model_requests', 'costs.tool_calls', 'costs.prompt_tokens', 'costs.eval_tokens']


def metric_value(scores, metric):
    """A float or None; None drops the scenario from that metric's pairs."""
    if metric == 'elapsed_s':
        return scores['costs']['elapsed_ms'] / 1000
    value = scores
    for part in metric.split('.'):
        value = value.get(part) if isinstance(value, dict) else None
    return None if value is None else float(value)


def paired_bootstrap(units, contrasts, *, seed=SEED, resamples=RESAMPLES):
    """units: [{id, stratum, values: {run: float}}], every run present; contrasts: [(left, right)]."""
    if not units:
        raise ValueError('No paired units.')
    units = sorted(units, key=lambda u: (u['stratum'], u['id']))
    if len({u['id'] for u in units}) != len(units):
        raise ValueError('A scenario appears twice.')
    names = sorted({a for u in units for a in u['values']})
    if any(set(u['values']) != set(names) for u in units):
        raise ValueError('A run is missing from a unit.')
    values = np.array([[u['values'][a] for a in names] for u in units], dtype=float)
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
              'means': {a: float(values[:, i].mean()) for i, a in enumerate(names)}, 'contrasts': []}
    for left, right in contrasts:
        if left not in names or right not in names:
            raise ValueError(f'Contrast names a run without results: {left} minus {right}')
        delta = values[:, names.index(left)] - values[:, names.index(right)]
        boot = means[:, names.index(left)] - means[:, names.index(right)]
        result['contrasts'].append({'comparison': f'{left} minus {right}', 'mean_difference': float(delta.mean()),
                                    'nominal_95_ci': np.quantile(boot, [.025, .975]).tolist(),
                                    'wins': int((delta > 1e-12).sum()), 'ties': int((abs(delta) <= 1e-12).sum()),
                                    'losses': int((delta < -1e-12).sum())})
    return result


def load_runs(runs):
    """runs: {name: path}; returns {name: {scenario_id: row}} and the run metadata."""
    rows, meta = {}, {}
    for name, path in runs.items():
        path = Path(path)
        rows[name] = {r['scenario']['id']: r for r in
                      (json.loads(line) for line in (path / 'results.jsonl').read_text().splitlines() if line.strip())}
        meta[name] = read_json(path / 'run.json')
    return rows, meta


def units_for(rows, metric, *, slices=None):
    """Scenarios shared by every run with the metric defined in every run, stratified by slice."""
    shared = sorted(set.intersection(*(set(r) for r in rows.values())))
    units = []
    for sid in shared:
        scenario = next(iter(rows.values()))[sid]['scenario']
        if slices and scenario['slice'] not in slices:
            continue
        values = {name: metric_value(rows[name][sid]['scores'], metric) for name in rows}
        if any(v is None for v in values.values()):
            continue
        units.append({'id': sid, 'stratum': scenario['slice'], 'values': values})
    return units


def analyze(runs, contrasts, *, seed=SEED, resamples=RESAMPLES):
    rows, meta = load_runs(runs)
    slices = sorted({r['scenario']['slice'] for name in rows for r in rows[name].values()})
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
    return {'runs': {name: {'label': m['label'], 'model': m['model'], 'think': m['think'], 'git_head': m['git_head'],
                            'dirty': m['dirty']} for name, m in meta.items()},
            'contrasts': [f'{a} minus {b}' for a, b in contrasts], 'seed': seed, 'resamples': resamples,
            'method': 'per-scenario paired differences; slice-stratified percentile bootstrap; nominal 95% intervals; '
                      'no multiplicity correction', 'blocks': blocks}


def _fmt(metric, value):
    if metric.endswith('tokens'):
        return f'{value:+,.0f}'
    if metric in ('elapsed_s', 'costs.model_requests', 'costs.tool_calls'):
        return f'{value:+.1f}'
    return f'{value:+.3f}'


def markdown(report):
    names = list(report['runs'])
    lines = ['# Paired comparison', '',
             'Development material, one run per configuration, nominal 95% percentile intervals, no multiplicity '
             f"correction. Seed {report['seed']}, {report['resamples']} resamples, stratified by slice; "
             'counts are wins/ties/losses for the left run; `*` marks an interval that excludes zero.', '']
    for name, m in report['runs'].items():
        lines.append(f"- {name}: `{m['label']}` ({m['model']}, think={m['think']}, git={m['git_head'][:10]}{'+dirty' if m['dirty'] else ''})")
    lines += ['', '| slice | metric | n | ' + ' | '.join(names) + ' | contrast | diff | 95% CI | W/T/L |',
              '| --- | --- | ---: | ' + ' | '.join('---:' for _ in names) + ' | --- | ---: | --- | --- |']
    for block in report['blocks']:
        metric = block['metric']
        means = ' | '.join(_fmt(metric, block['means'][a]).lstrip('+') for a in names)
        for i, c in enumerate(block['contrasts']):
            head = f"| {block['slice']} | {metric.replace('behavior.', '').replace('costs.', '')} | {block['n']} | {means} |" if i == 0 \
                else '| | | | ' + ' | '.join('' for _ in names) + ' |'
            lo, hi = c['nominal_95_ci']
            star = '*' if (lo > 0 or hi < 0) else ''
            lines.append(f"{head} {c['comparison']} | {_fmt(metric, c['mean_difference'])} | [{_fmt(metric, lo)}, {_fmt(metric, hi)}]{star} | "
                         f"{c['wins']}/{c['ties']}/{c['losses']} |")
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='append', required=True, help='NAME=path to a run directory; repeatable')
    p.add_argument('--contrast', action='append', help='LEFT:RIGHT; repeatable; default: every later run minus the first')
    p.add_argument('--output', type=Path, help='Markdown destination; a .json sibling is written beside it')
    p.add_argument('--resamples', type=int, default=RESAMPLES)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    runs = dict(item.split('=', 1) for item in args.run)
    names = list(runs)
    contrasts = [tuple(item.split(':', 1)) for item in args.contrast] if args.contrast else [(n, names[0]) for n in names[1:]]
    report = analyze(runs, contrasts, seed=args.seed, resamples=args.resamples)
    text = markdown(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        args.output.with_suffix('.json').write_text(json.dumps(report, indent=1) + '\n')
    print(text)


if __name__ == '__main__':
    main()

"""Outcome-blind precision, allocation, execution-policy and cost planning for core v2.

Nothing here reads answers, judge labels or observed quality. Selection uses
frozen ID hash order only; exposed development questions are excluded by ID.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import NormalDist

from arkb.evaluation.external import digest, write_json
from evaluation.agentic_tools.contract import ARMS
from evaluation.agentic_tools.common import utc
from evaluation.agentic_tools.records import read_records
from evaluation.agentic_tools.selection import key

COUNTS = {'browsecomp-plus': 320, 'fiqa': 50, 'nfcorpus': 50, 'musique': 30}
PRIMARY_REPETITIONS = (0,)
REPEAT_SUBSET = {'dataset': 'browsecomp-plus', 'questions': 20, 'repetitions': (0, 1, 2)}
PRIMARY_TOTAL = 3360
REPEAT_TOTAL = 280
TOTAL = PRIMARY_TOTAL + REPEAT_TOTAL
TIME_CAP_HOURS = 32
PRIMARY_FAMILY = 4
CORE_OPERATIONAL_POLICY = {
    'schema': 'arkb-core-operational-policy-v1',
    'concurrency': 1,
    'serving_basis': 'serving-v1 negative client-concurrency screen; qwen35 has one execution slot in Ollama 0.33.2',
    'tool_outcomes': 'timeouts, tool errors, budget stops, invalid finals and nonanswers are measured outcomes retained in every denominator; never rerun, replaced or excluded',
    'scope_fixtures': 'frozen v3 readiness workload runs before the first measured attempt of every scope; a failed fixture pauses the run',
    'pause_on_execution_failure': True,
    'pause_on_gpu_competition': True,
    'max_operational_failures_per_attempt': 3,
    'window_min_valid_calls': 200,
    'window_max_operational_rate': 0.01,
    'review_rule': 'operational-review.json lists reviewed attempt keys; reviewed attempts stay in the data, leave the pause rules and restart the rate window',
    'trajectory_time_cap_hours': TIME_CAP_HOURS,
    'time_cap_rule': 'measured trajectory seconds of completed attempts; finish and preserve the current attempt, then pause; extension only by an explicit recorded time-cap-extension.json',
    'fixed_rag_retrieval': 'repeated on every attempt as in pilot v3; no cache',
    'automatic_retries': False,
    'resume': 'verify all identities, skip immutable completed attempts, account explicitly for any interrupted attempt',
}


def selection_for_core(original, exposed):
    selected = deepcopy(original)
    selected.update(schema='agentic-tools-selection-core-v2', repetitions={'primary': list(PRIMARY_REPETITIONS),
                    'repeat_subset': {**REPEAT_SUBSET, 'repetitions': list(REPEAT_SUBSET['repetitions'])}},
                    exclusion_policy='Exclude all pilot IDs and every started old-core question; keep the frozen SHA order.')
    for dataset, strata in selected['tracks'].items():
        for stratum in strata:
            population = stratum['pilot'] + stratum['core'] + stratum['reserves']
            if len(population) != len(set(population)) or len(population) != stratum['population_count']:
                raise ValueError('Invalid frozen population.')
            excluded = set(stratum['pilot']) | set(exposed.get(dataset, []))
            ordered = sorted((x for x in population if x not in excluded), key=lambda x: (key(dataset, stratum['stratum'], x), x))
            count = 10 if dataset == 'musique' else COUNTS[dataset]
            if len(ordered) < count:
                raise ValueError('Insufficient unexposed questions.')
            stratum.update(pilot=[], core=ordered[:count], reserves=ordered[count:],
                           development_excluded=sorted(set(population) & excluded), eligible_count=len(ordered))
    return selected


def repeat_ids(selection):
    stratum = selection['tracks'][REPEAT_SUBSET['dataset']][0]
    return stratum['core'][:REPEAT_SUBSET['questions']]


def core_schedule(selection):
    repeated = set(repeat_ids(selection))
    result, unit = [], 0
    for dataset, strata in selection['tracks'].items():
        for stratum in strata:
            for identifier in stratum['core']:
                repetitions = (REPEAT_SUBSET['repetitions'] if dataset == REPEAT_SUBSET['dataset'] and identifier in repeated
                               else PRIMARY_REPETITIONS)
                for repetition in repetitions:
                    offset = (unit + repetition) % len(ARMS)
                    for arm in ARMS[offset:] + ARMS[:offset]:
                        for variant in (('v0', 'v1') if dataset == 'musique' else ('v0',)):
                            result.append({'dataset': dataset, 'id': identifier, 'stratum': stratum['stratum'],
                                           'variant': variant, 'arm': arm.id, 'repetition': repetition, 'unit_index': unit})
                unit += 1
    return result


def exposed_questions(old_core):
    exposed = defaultdict(set)
    for start in (Path(old_core) / 'core-attempts').glob('*/attempt.json'):
        s = json.loads(start.read_text())['schedule']
        exposed[s['dataset']].add(s['id'])
    return {ds: sorted(ids) for ds, ids in exposed.items()}


def family_z(alpha=.05, family=PRIMARY_FAMILY):
    return NormalDist().inv_cdf(1 - alpha / family / 2)


def required_questions(half_width, discordance, *, delta=0.0, alpha=.05, family=PRIMARY_FAMILY):
    z = family_z(alpha, family)
    return math.ceil(z * z * (discordance - delta ** 2) / half_width ** 2)


def precision(n, repetitions, discordance, rho, population=825, delta=.05):
    if not (0 < n <= population and repetitions >= 1 and 0 <= rho <= 1 and abs(delta) <= discordance <= 1):
        raise ValueError('Invalid precision scenario.')
    variance = discordance - delta ** 2
    z = family_z()
    factor = rho + (1 - rho) / repetitions
    se = math.sqrt(variance * factor / n)
    finite_se = math.sqrt(variance * (rho * (population - n) / (population - 1) + (1 - rho) / repetitions) / n)
    power = 1 - NormalDist().cdf(z - delta / se) + NormalDist().cdf(-z - delta / se) if se else 1.0
    return {'questions': n, 'repetitions': repetitions, 'discordance': discordance, 'paired_difference_rho': rho,
            'assumed_delta': delta, 'effective_n': n / factor, 'normal_half_width': z * se,
            'normal_power_at_assumed_delta': power, 'finite_expected_population_half_width': z * finite_se,
            'finite_population': population}


def plan(pilot, old_core, destination):
    protocol, rows = read_records(pilot)
    original = json.loads((pilot / 'selection.json').read_text())
    exposed = exposed_questions(old_core)
    selection = selection_for_core(original, exposed)
    schedule = core_schedule(selection)
    if len(schedule) != TOTAL or sum(r['repetition'] == 0 for r in schedule) != PRIMARY_TOTAL:
        raise ValueError('Unexpected replacement schedule size.')
    costs = defaultdict(list)
    for row in rows:
        costs[row['schedule']['dataset']].append(row['elapsed_ms'] / 1000)
    counts = defaultdict(int)
    for row in schedule:
        counts[row['dataset']] += 1
    expected = {ds: sum(values) / len(values) * counts[ds] for ds, values in costs.items()}
    hours = sum(expected.values()) / 3600
    population = selection['tracks']['browsecomp-plus'][0]['eligible_count']
    result = {'schema': 'agentic-core-v2-precision-cost-plan', 'created_at': utc(),
              'planning_source_sha256': digest(__file__), 'pilot_protocol_sha256': digest(pilot / 'protocol.json'),
              'pilot_accounting_sha256': digest(pilot / 'pilot-accounting.json'),
              'old_core_protocol_sha256': digest(old_core / 'protocol.json'),
              'exposed_old_core': exposed, 'counts': COUNTS,
              'attempts': {'total': TOTAL, 'primary': PRIMARY_TOTAL, 'repeat_subset': REPEAT_TOTAL,
                           'by_dataset': dict(counts)},
              'primary_repetitions': list(PRIMARY_REPETITIONS),
              'repeat_subset': {'dataset': REPEAT_SUBSET['dataset'], 'questions': REPEAT_SUBSET['questions'],
                                'repetitions': list(REPEAT_SUBSET['repetitions']), 'ids': repeat_ids(selection),
                                'selection_rule': 'first registered core IDs in frozen hash order; the same prefix as the citation-review sample',
                                'estimand': 'between-session variability of answer success and of the four primary paired differences under the identical registered protocol, environment and adjacent scheduling; one-way random-effects decomposition on 20 questions; descriptive only',
                                'exclusions': 'repetitions one and two never enter primary contrasts, evidence comparisons, cost ratios or the citation review; no best-of-k'},
              'estimands': {
                  'single_run': 'mean single-session outcome of each arm over the registered question distribution, including session variability, on repetition zero of every registered question; paired A-All minus restricted-arm differences on the same questions',
                  'repeat_subset': 'within-question between-session variance components on the registered subset; not additional independent questions'},
              'selection': selection,
              'design_target': {'primary_family': PRIMARY_FAMILY, 'alpha_per_comparison': .05 / PRIMARY_FAMILY,
                                'half_width_at_discordance_at_most_half': .10,
                                'normal_required_n_conservative_delta_zero': required_questions(.10, .5),
                                'chosen_n': COUNTS['browsecomp-plus'], 'minimum_observed_gain': .05,
                                'guaranteed_detection_of_five_points': False,
                                'rationale': 'One repetition maximizes independent questions for any repeat correlation: effective n equals n divided by rho plus (1 minus rho) over repetitions, so 320 single sessions beat 100 triple sessions under every scenario below. Precision, not power at five points, is the registered target.'},
              'precision_scenarios': [precision(n, r, q, rho, population, delta)
                  for n, r in [(100, 3), (200, 1), (320, 1), (500, 1), (population, 1)]
                  for q in (.1, .25, .5, 1.) for rho in (0., .25, .5, .75, 1.) for delta in (.05, .10)],
              'execution_policy': CORE_OPERATIONAL_POLICY,
              'cost': {'pilot_seconds_mean_by_dataset': {ds: sum(v) / len(v) for ds, v in costs.items()},
                       'expected_trajectory_seconds_by_dataset': expected,
                       'expected_trajectory_hours': hours,
                       'planning_hours_with_25_percent_headroom': hours * 1.25,
                       'trajectory_time_cap_hours': TIME_CAP_HOURS,
                       'cap_policy': CORE_OPERATIONAL_POLICY['time_cap_rule'],
                       'excluded': ['scope setup and fixtures', 'judge inference', 'human review', 'administrative waiting', 'archival'],
                       'uncertainty': 'Small v3 pilot extrapolation, not a confidence interval or completion guarantee.'},
              'quality_scores_used': False, 'core_dispatch_authorized': False}
    destination.mkdir(exist_ok=True, parents=True)
    write_json(destination / 'precision-cost.json', result)
    write_json(destination / 'selection.json', selection)
    write_json(destination / 'core-schedule.json', schedule)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--old-core', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.pilot, args.old_core, args.destination)
    print(json.dumps({k: result[k] for k in ('counts', 'attempts', 'exposed_old_core', 'design_target', 'cost')}, indent=2))


if __name__ == '__main__':
    main()

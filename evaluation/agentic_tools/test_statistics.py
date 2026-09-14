import pytest

from .contract import ARMS
from .statistics import paired_comparisons, query_units


def test_query_means_do_not_treat_repetitions_as_questions():
    rows = [{'schedule': {'id': str(q), 'arm': a.id, 'stratum': 'all', 'variant': 'v0', 'repetition': rep},
             'score': float(rep == 0)} for q in range(2) for a in ARMS for rep in range(3)]
    units = query_units(rows, 'score')
    assert len(units) == 2 and units[0]['values']['A-All'] == 1/3
    result = paired_comparisons(units, primary=True, resamples=100)
    assert result['independent_units'] == 2
    assert all(c['mean_difference'] == 0 and not c['meaningful_gain_demonstrated'] for c in result['contrasts'][:4])
    with pytest.raises(ValueError, match='missing a repetition'):
        query_units(rows[:-1], 'score')


def test_pending_judge_labels_block_definitive_comparisons():
    units = [{'id': 'x', 'stratum': 'all', 'values': {a.id: None if a.id == 'A-All' else 0 for a in ARMS}}]
    assert paired_comparisons(units, primary=True)['status'] == 'pending_scores'


def test_paired_stratified_resampling_preserves_hop_groups():
    units = [{'id': str(i), 'stratum': str(i % 3 + 2),
              'values': {a.id: float(a.id == 'A-All') for a in ARMS}} for i in range(30)]
    result = paired_comparisons(units, primary=True, resamples=100)
    assert result['stratum_sizes'] == {'2': 10, '3': 10, '4': 10}
    assert result['advantage_over_every_restricted_arm']
    assert result['contrasts'][0]['adjusted_98_75_ci'] == [1, 1]

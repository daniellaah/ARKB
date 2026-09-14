import pytest

from .analyze import musique_pairs, citation_sample
from .contract import ARMS


def test_pair_sufficiency_needs_both_variants_and_keeps_repetitions():
    rows = []
    for rep in range(3):
        for variant in range(2):
            rows.append({'schedule': {'dataset': 'musique', 'id': 'q', 'arm': 'A-All', 'repetition': rep, 'variant': f'v{variant}'},
                         'musique': {'gold_answerable': variant == 0, 'answerability_correct': float(variant == 0),
                                     'answer_f1': 1 if variant == 0 else None, 'answer_em': 1 if variant == 0 else None,
                                     'support_f1': 1 if variant == 0 else None}})
    pairs = musique_pairs(rows)
    assert len(pairs) == 3 and all(p['group_answer_sufficiency_f1'] == 0 for p in pairs)
    with pytest.raises(ValueError, match='incomplete'):
        musique_pairs(rows[:-1])


def test_citation_export_is_complete_and_hides_arm_identity():
    ids = [str(i) for i in range(20)]
    rows = [{'key': f'{q}-{a.id}', 'scenario_id': q, 'schedule': {'dataset': 'browsecomp-plus', 'id': q,
              'arm': a.id, 'repetition': 0}, 'result': None} for q in ids for a in ARMS]
    review, key = citation_sample(rows, ids, {q: {'query': 'synthetic'} for q in ids})
    assert len(review) == len(key) == 140
    assert all('arm' not in r and 'schedule' not in r and 'expected_answer' not in r for r in review)
    with pytest.raises(ValueError, match='140|20 complete'):
        citation_sample([rows[0], *rows[:-1]], ids, {q: {'query': 'synthetic'} for q in ids})

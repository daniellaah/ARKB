from dataclasses import asdict

import pytest

from arkb.agent.state import AgentFinal
from .scoring import answer_eligible, musique_row, evidence_coverage, costs_and_behavior


def row(status='answered', answer='The Seven', citations=None):
    return {'key': 'fixture', 'schedule': {'id': 'q', 'arm': 'A-All'}, 'elapsed_ms': 1,
            'result': {'final': asdict(AgentFinal(answer, status, citations or [], 'fixture'))}}


GOLD = {'id': 'q', 'answer': 'seven', 'answer_aliases': ['7'], 'answerable': True,
        'source_map': {'s.md': 3}, 'support_idxs': [3]}


def test_failure_cannot_earn_abstention_credit_or_use_its_answer():
    failed = row('error', 'seven')
    assert not answer_eligible(failed)
    metric = musique_row(failed, {**GOLD, 'answerable': False})
    assert metric['answerability_correct'] == 0 and not metric['execution_valid']
    assert musique_row(failed, GOLD)['answer_f1'] == 0


def test_partial_is_false_answerability_without_repairing_prose():
    r = row('partial', 'seven', [{'source': 's.md'}])
    assert answer_eligible(r)
    metric = musique_row(r, GOLD)
    assert metric['prediction']['predicted_answerable'] is False
    assert metric['answer_f1'] == 1 and metric['answerability_correct'] == 0
    assert metric['support_f1'] == 1


def test_unknown_support_source_is_not_silently_discarded():
    with pytest.raises(ValueError, match='outside'):
        musique_row(row(citations=[{'source': 'unknown.md'}]), GOLD)


def test_insufficient_is_nonanswer_even_when_prose_contains_gold():
    assert not answer_eligible(row('insufficient_evidence', 'seven'))


def test_evidence_stages_and_unjudged_sources_stay_distinct():
    r = {**row(), 'evidence_sources': {'returned': ['a.md', 'b.md'], 'delivered': ['b.md'], 'submitted': []}}
    labels = {'source_map': {'a.md': 'a', 'b.md': 'b'}, 'qrels': {'q': {'a': 1}}}
    scores = evidence_coverage(r, labels)
    assert scores['qrels_returned']['positive_recall'] == 1
    assert scores['qrels_delivered']['positive_recall'] == 0
    assert scores['qrels_delivered']['unjudged_sources'] == 1
    assert scores['qrels_submitted']['observed_sources'] == 0


def test_hybrid_work_has_two_completed_legs_and_failed_work_is_unknown():
    r = row()
    r['result']['observation'] = {'tools': [
        {'executed': True, 'name': 'search', 'arguments': {'mode': 'hybrid'}, 'status': 'success', 'raw_result': {'results': []}},
        {'executed': True, 'name': 'search', 'arguments': {'mode': 'semantic'}, 'status': 'fatal_error'},
    ]}
    costs = costs_and_behavior(r)
    assert costs['completed_retrieval_legs'] == {'bm25': 1, 'semantic': 1}
    assert costs['search_calls_with_unknown_partial_leg_work'] == 1

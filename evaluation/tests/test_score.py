"""Scorer checks on synthetic records; labels are used only for scoring, never for selection."""
from pathlib import Path

import pytest

from evaluation.common import ROOT
from evaluation.score import (PRIMARY, cited_sources, costs, delivered_observations, delivered_sources, first_line,
                              markdown, match_limit_hits, musique_scores, score_row, summarize)
from evaluation.v2 import load_dataset


def record(tools, final=None, refs=None, models=None):
    return {'elapsed_ms': 1000, 'error': None, 'result': {
        'final': {'termination_reason': 'final', **(final or {'status': 'answered', 'answer': 'x', 'citations': []})}, 'stop_reason': 'final',
        'observation': {'models': models or [{'usage': {'prompt_eval_count': 10, 'eval_count': 5}}], 'tools': tools,
                        'evidence_references': refs or {}, 'evidence': {'delivered_tokens': 3},
                        'usage': {'prompt_eval_count': {'known_total': 10}, 'eval_count': {'known_total': 5}}}}}


REFS = {'ev_1': {'source': 'a.md', 'document_revision': 'r', 'start_char': 0, 'end_char': 3, 'content': 'abc'},
        'ev_2': {'source': 'b.md', 'document_revision': 'r', 'start_char': 0, 'end_char': 3, 'content': 'abc'}}


def tool(name, hits, *, delivered=True, delivered_refs=None, arguments=None):
    key = 'result' if name == 'read' else 'results'
    return {'name': name, 'executed': True, 'status': 'success', 'arguments': arguments or {},
            'raw_result': {key: hits[0] if name == 'read' else hits}, 'delivered_to_conversation': delivered,
            'submitted_to_model': False, **({'delivered_refs': delivered_refs} if delivered_refs is not None else {})}


def test_delivered_evidence_and_costs_from_the_trace():
    tools = [tool('match', [{'ref': 'ev_1', 'source': 'a.md'}, {'ref': 'ev_2', 'source': 'b.md'}], arguments={'query': 'abc', 'limit': 2},
                  delivered_refs=['ev_1']),
             tool('read', [{'ref': 'ev_2', 'source': 'b.md'}], delivered=False)]
    row = record(tools, refs=REFS, models=[{'response': {'done_reason': 'length'}}, {'response': {'done_reason': 'stop'}}])
    report = row['result']['observation']
    assert [o['source'] for o in delivered_observations(report)] == ['a.md']
    assert delivered_sources(report) == ['a.md']
    assert match_limit_hits(row) == 1
    c = costs(row)
    assert c['tool_calls'] == 2 and c['tools_executed'] == {'match': 1, 'read': 1} and c['responses_cut'] == 1
    assert c['prompt_tokens'] == 10 and c['delivered_evidence_tokens'] == 3 and c['model_requests'] == 2
    row['result']['final']['citations'] = [{'ref': 'ev_2', 'source': 'b.md'}]
    assert cited_sources(row) == ['b.md']


def test_exact_scoring_counts_cited_completeness_and_spurious_sources():
    tools = [tool('match', [{'ref': 'ev_1', 'source': 'a.md'}], arguments={'query': 'abc'})]
    row = record(tools, final={'status': 'answered', 'answer': 'a', 'citations': [{'ref': 'ev_1', 'source': 'a.md'}, {'ref': 'x', 'source': 'z.md'}]}, refs=REFS)
    scored = score_row(row, {'id': 'x', 'slice': 'exact-v2', 'stratum': 'df_2_5'}, {'expected_sources': ['a.md', 'b.md']}, {})
    assert scored['completeness_cited'] == .5 and scored['completeness_delivered'] == .5
    assert scored['spurious_cited'] == 1 and scored['complete_and_exact_cited'] is False and scored['match_calls'] == 1
    assert scored['error'] is None and scored['final_status'] == 'answered'


def test_recall_scoring_maps_delivered_sources_to_positive_documents():
    tools = [tool('search', [{'ref': 'ev_1', 'source': 'a.md'}, {'ref': 'ev_2', 'source': 'b.md'}])]
    resources = {'source_maps': {'nfcorpus': {'a.md': 'D1', 'b.md': 'D2'}}}
    labels = {'qrels': {'D1': 2, 'D3': 1, 'D4': 0}}
    scored = score_row(record(tools, refs=REFS), {'id': 'r', 'slice': 'recall-nfcorpus', 'scope': 'nfcorpus'}, labels, resources)
    assert scored['positive_recall_delivered'] == .5 and scored['answered'] is True
    with pytest.raises(ValueError, match='outside'):
        score_row(record(tools, refs=REFS), {'id': 'r', 'slice': 'recall-nfcorpus', 'scope': 'nfcorpus'}, labels, {'source_maps': {'nfcorpus': {'a.md': 'D1'}}})


def test_v2_scoring_uses_span_labels_and_behavior_rules():
    dataset = load_dataset(ROOT / 'evaluation/devset/v2-pilot', allow_provisional=True)
    resources = {'v2': dataset, 'source_maps': {}}
    case = next(c for c in dataset.cases if c['task_type'] == 'semantic_discovery')
    span = dataset.evidence[case['evidence_requirements'][0]['alternatives'][0][0]]
    refs = {'ev_1': {'source': span['source'], 'document_revision': span['document_revision'], 'start_char': None,
                     'end_char': None, 'content': dataset.bodies[span['source']]}}
    tools = [tool('read', [{'ref': 'ev_1', 'source': span['source']}])]
    final = {'status': 'answered', 'answer': 'x', 'citations': [{'ref': 'ev_1', 'source': span['source']}]}
    scored = score_row(record(tools, final=final, refs=refs), {'id': case['id'], 'slice': 'v2', 'task_type': case['task_type']}, {}, resources)
    assert scored['evidence_coverage_delivered'] == 1.0 and scored['critical_covered_delivered'] is True
    assert scored['source_recall_cited'] == 1.0 and scored['behavior']['answered'] is True
    quiet = score_row(record([]), {'id': case['id'], 'slice': 'v2', 'task_type': 'no_retrieval'}, {}, resources)
    assert quiet['behavior']['no_retrieval_respected'] is True and quiet['evidence_coverage_delivered'] == 0.0


def test_musique_first_line_scores_and_error_finals():
    gold = {'id': 'p1', 'answerable': True, 'answer': 'Paris', 'answer_aliases': [], 'support_idxs': [0, 1], 'source_map': {'a.md': 0, 'c.md': 2}}
    row = record([], final={'status': 'answered', 'answer': '\nParis\nThe capital named in the passage is Paris.', 'citations': [{'ref': 'ev_1', 'source': 'a.md'}]})
    scored = musique_scores(row, gold)
    assert scored['answer_em_first_line'] == 1.0 and scored['answer_f1_first_line'] == 1.0
    assert scored['support_f1'] == pytest.approx(2 / 3) and scored['answerability_correct'] == 1.0
    assert first_line('  \n\n  Paris  \nmore') == 'Paris' and first_line(None) == ''
    unanswerable = musique_scores(row, {**gold, 'answerable': False})
    assert unanswerable['answer_f1_first_line'] is None and unanswerable['answerability_correct'] == 0.0
    failed = musique_scores({**row, 'result': None, 'error': {'type': 'X', 'message': 'y'}}, gold)
    assert failed['answer_f1_first_line'] == 0.0 and failed['support_f1'] == 0.0 and failed['answerability_correct'] == 0.0
    with pytest.raises(ValueError, match='outside'):
        musique_scores(record([], final={'status': 'answered', 'answer': 'x', 'citations': [{'ref': 'r', 'source': 'zz.md'}]}), gold)


def test_summary_and_markdown(tmp_path):
    rows = []
    for i, value in enumerate((0.0, 1.0, 1.0)):
        row = record([], final={'status': 'answered', 'answer': 'x', 'citations': []})
        row['scenario'] = {'id': f'e{i}', 'slice': 'exact-v2', 'stratum': 'df_2_5'}
        row['scores'] = {**score_row(row, row['scenario'], {'expected_sources': ['a.md']}, {}), 'completeness_cited': value}
        rows.append(row)
    summary = summarize(rows)
    block = summary['exact-v2']
    assert block['scenarios'] == 3 and block['completeness_cited'] == pytest.approx(2 / 3) and block['errors'] == 0
    assert block['responses_cut'] == 0 and 'match_limit_hits' in block
    text = markdown(summary, {'label': 'x', 'model': 'm', 'think': False, 'git_head': 'abcdef0123', 'dirty': False, 'scenarios': 3, 'wall_seconds': 1.0})
    assert '| exact-v2 | 3 | completeness_cited | 0.667 |' in text and PRIMARY['musique'] == 'answer_f1_first_line'
    assert Path(__file__).exists()

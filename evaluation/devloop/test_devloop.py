"""Development-loop builder and scorer checks; no services, no answers used for selection."""
import json

import pytest

from arkb.evaluation.external import write_json
from evaluation.devloop.build import exact_tasks, term_key
from evaluation.devloop.run import (PRIMARY, DevChatClient, cited_sources, compare, first_line, match_limit_hits, score_row,
                                    stage_observations, summarize, markdown, load_resources)


def corpus(tmp_path):
    root = tmp_path / 'corpus'
    root.mkdir()
    texts = {'a.md': '# A\n\nQuantum lattice studies mention magnesium twice: magnesium.',
             'b.md': '# B\n\nMagnesium and lattice appear here; quantum too.',
             'c.md': '# C\n\nOnly lattice appears in this one, with tokens galore.',
             'd.md': '# D\n\nNothing shared except lattice and tokens galore again.'}
    for name, text in texts.items():
        (root / name).write_text(text)
    return root


def test_exact_tasks_truth_is_substring_truth_and_deterministic(tmp_path):
    root = corpus(tmp_path)
    strata = {'df_2_5': (2, 5, 2)}
    tasks = exact_tasks(root, 'fixture', strata=strata, phrase=(2, 10, 1))
    assert len(tasks) == 3
    for task in tasks:
        term = task['term']
        expected = sorted(n for n in ('a.md', 'b.md', 'c.md', 'd.md') if term in (root / n).read_text().split('\n', 2)[2])
        assert task['labels']['expected_sources'] == expected
        assert 2 <= len(expected) <= 10
    assert [t['term'] for t in tasks] == [t['term'] for t in exact_tasks(root, 'fixture', strata=strata, phrase=(2, 10, 1))]
    assert term_key('fixture', 'lattice') != term_key('other', 'lattice')
    with pytest.raises(ValueError, match='Insufficient'):
        exact_tasks(root, 'fixture', strata={'df_2_5': (2, 5, 50)}, phrase=(2, 10, 0))


def record(tools, final=None, refs=None):
    return {'elapsed_ms': 1000, 'error': None, 'result': {
        'final': {'termination_reason': 'final', **(final or {'status': 'answered', 'answer': 'x', 'citations': []})}, 'stop_reason': 'final',
        'observation': {'models': [{'usage': {'prompt_eval_count': 10, 'eval_count': 5}}], 'tools': tools,
                        'evidence_references': refs or {}, 'evidence': {'delivered_tokens': 3}, 'usage': None}}}


def test_stage_observations_and_limit_hits():
    refs = {'ev_1': {'source': 'a.md', 'document_revision': 'r', 'start_char': 0, 'end_char': 3, 'content': 'abc'},
            'ev_2': {'source': 'b.md', 'document_revision': 'r', 'start_char': 0, 'end_char': 3, 'content': 'abc'}}
    tools = [{'name': 'match', 'executed': True, 'status': 'success', 'arguments': {'query': 'abc', 'limit': 2},
              'raw_result': {'results': [{'ref': 'ev_1', 'source': 'a.md'}, {'ref': 'ev_2', 'source': 'b.md'}]}, 'delivered_to_conversation': True, 'submitted_to_model': False},
             {'name': 'read', 'executed': True, 'status': 'success', 'arguments': {'ref': 'ev_1'},
              'raw_result': {'result': {'ref': 'ev_1', 'source': 'a.md'}}, 'delivered_to_conversation': False, 'submitted_to_model': False}]
    row = record(tools, refs=refs)
    stages = stage_observations(row['result']['observation'])
    assert [o['source'] for o in stages['returned']] == ['a.md', 'b.md', 'a.md']
    assert [o['source'] for o in stages['delivered']] == ['a.md', 'b.md'] and stages['submitted'] == []
    assert match_limit_hits(row) == 1
    row['result']['final']['citations'] = [{'ref': 'ev_2', 'source': 'b.md'}]
    assert cited_sources(row) == ['b.md']


def test_exact_scoring_counts_completeness_per_stage():
    refs = {'ev_1': {'source': 'a.md', 'document_revision': 'r', 'start_char': 0, 'end_char': 3, 'content': 'abc'}}
    tools = [{'name': 'match', 'executed': True, 'status': 'success', 'arguments': {'query': 'abc'},
              'raw_result': {'results': [{'ref': 'ev_1', 'source': 'a.md'}]}, 'delivered_to_conversation': True, 'submitted_to_model': True}]
    row = record(tools, final={'status': 'answered', 'answer': 'a', 'citations': [{'ref': 'ev_1', 'source': 'a.md'}]}, refs=refs)
    scenario = {'id': 'x', 'slice': 'exact-nfcorpus', 'stratum': 'df_2_5'}
    scored = score_row(row, scenario, {'expected_sources': ['a.md', 'b.md']}, {})
    assert scored['completeness_returned'] == .5 and scored['completeness_cited'] == .5
    assert scored['match_calls'] == 1 and scored['complete_and_exact_cited'] is False
    b = score_row(record(tools, final={'status': 'answered', 'answer': 'different text', 'citations': [{'ref': 'ev_1', 'source': 'a.md'}]}, refs=refs),
                  scenario, {'expected_sources': ['a.md', 'b.md']}, {})
    assert {k: v for k, v in b.items() if k != 'costs'} == {k: v for k, v in scored.items() if k != 'costs'}


def test_v2_scoring_uses_span_labels_and_behavior_rules():
    resources = load_resources(__import__('pathlib').Path('evaluation/devloop/devset-v1')) if False else \
        {'v2': __import__('arkb.evaluation.v2', fromlist=['load_dataset']).load_dataset(
            __import__('pathlib').Path('evaluation/data/v2/pilot'), notes_dir=__import__('pathlib').Path('evaluation/data/v2/pilot/corpus'), allow_provisional=True),
         'source_maps': {}}
    dataset = resources['v2']
    case = next(c for c in dataset.cases if c['task_type'] == 'semantic_discovery')
    span = dataset.evidence[case['evidence_requirements'][0]['alternatives'][0][0]]
    body = dataset.bodies[span['source']]
    refs = {'ev_1': {'source': span['source'], 'document_revision': span['document_revision'], 'start_char': None,
                     'end_char': None, 'content': body}}
    tools = [{'name': 'read', 'executed': True, 'status': 'success', 'arguments': {'source': span['source']},
              'raw_result': {'result': {'ref': 'ev_1', 'source': span['source']}}, 'delivered_to_conversation': True, 'submitted_to_model': True}]
    scenario = {'id': case['id'], 'slice': 'v2', 'task_type': case['task_type']}
    scored = score_row(record(tools, refs=refs), scenario, {}, resources)
    assert scored['evidence_coverage_delivered'] == 1.0 and scored['critical_covered_submitted'] is True
    assert scored['source_recall_returned'] == 1.0
    quiet = score_row(record([]), {'id': case['id'], 'slice': 'v2', 'task_type': 'no_retrieval'}, {}, resources)
    assert quiet['behavior']['no_retrieval_respected'] is True and quiet['evidence_coverage_delivered'] == 0.0


def test_summary_compare_and_markdown(tmp_path):
    def rows(value):
        out = []
        for i in range(3):
            row = record([], final={'status': 'answered', 'answer': 'x', 'citations': []})
            row['scenario'] = {'id': f'e{i}', 'slice': 'exact-nfcorpus', 'stratum': 'df_2_5'}
            row['scores'] = {**score_row(row, row['scenario'], {'expected_sources': ['a.md']}, {}), 'completeness_cited': value}
            out.append(row)
        return out
    a, b = rows(0.0), rows(1.0)
    summary = summarize(a)
    assert summary['exact-nfcorpus']['scenarios'] == 3 and summary['exact-nfcorpus']['completeness_cited'] == 0.0
    text = markdown(summary, {'label': 'x', 'adapter': 'product', 'model': 'm', 'think': False, 'git_head': 'abcdef0123', 'dirty': False,
                              'scenarios': 3, 'wall_seconds': 1.0})
    assert '| exact-nfcorpus | 3 | completeness_cited | 0.000 |' in text
    for name, group in (('a', a), ('b', b)):
        d = tmp_path / name
        d.mkdir()
        (d / 'results.jsonl').write_text('\n'.join(json.dumps(r) for r in group) + '\n')
        write_json(d / 'run.json', {'label': name, 'adapter': 'product', 'model': 'm', 'think': False, 'git_head': 'abcdef0123', 'dirty': False})
    result, text = compare(tmp_path / 'a', tmp_path / 'b', tmp_path / 'cmp.md')
    assert result['exact-nfcorpus:completeness_cited'] == {'a': 0.0, 'b': 1.0, 'delta': 1.0, 'wins': 3, 'ties': 0, 'losses': 0, 'n': 3}
    assert (tmp_path / 'cmp.md').exists() and PRIMARY['exact-nfcorpus'] == 'completeness_cited'


def test_first_line_answer_scores_take_the_short_answer_and_keep_full_text_scores():
    gold = {'id': 'p1', 'answerable': True, 'answer': 'Paris', 'answer_aliases': [], 'support_idxs': [0], 'source_map': {}}
    row = record([], final={'status': 'answered', 'answer': '\nParis\nThe capital named in the passage is Paris.', 'citations': []})
    scored = score_row(row, {'id': 'm', 'slice': 'musique', 'pair_id': 'p', 'variant': 'v0'}, {'gold': gold}, {})
    assert scored['answer_em_first_line'] == 1.0 and scored['answer_f1_first_line'] == 1.0
    assert scored['answer_em'] == 0.0 and 0 < scored['answer_f1'] < 1
    assert first_line('  \n\n  Paris  \nmore') == 'Paris' and first_line(None) == ''
    unanswerable = score_row(row, {'id': 'm', 'slice': 'musique', 'pair_id': 'p', 'variant': 'v1'}, {'gold': {**gold, 'answerable': False}}, {})
    assert unanswerable['answer_f1_first_line'] is None and unanswerable['answer_em_first_line'] is None
    failed = {**row, 'result': None, 'error': {'type': 'X', 'message': 'y'}}
    assert score_row(failed, {'id': 'm', 'slice': 'musique', 'pair_id': 'p', 'variant': 'v0'}, {'gold': gold}, {})['answer_f1_first_line'] == 0.0


class FakeHttp:
    def __init__(self):
        self.sent = []

    def post(self, path, *, json):
        self.sent.append(json)
        body = {'model': json['model'], 'done': True, 'done_reason': 'stop', 'message': {'role': 'assistant', 'content': '{}'}}
        return type('R', (), {'raise_for_status': lambda self: None, 'json': lambda self: body})()


def test_dev_client_keeps_the_callers_think_flag_and_rejects_another_model():
    client = DevChatClient.__new__(DevChatClient)
    client.model, client.options, client.think, client.http = 'qwen3.5:9b', {'num_ctx': 32768, 'num_predict': 4096}, True, FakeHttp()
    client.chat(model='qwen3.5:9b', messages=[], think=False, options={'temperature': 0})
    client.chat(model='qwen3.5:9b', messages=[])
    sent = client.http.sent
    assert [r['think'] for r in sent] == [False, True]
    assert sent[0]['options'] == {'temperature': 0, 'num_ctx': 32768, 'num_predict': 4096}
    assert all(r['truncate'] is False and r['shift'] is False for r in sent)
    with pytest.raises(ValueError, match='differs'):
        client.chat(model='qwen3.5:4b', messages=[])

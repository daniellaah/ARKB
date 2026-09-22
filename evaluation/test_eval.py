"""Question file validity and scoring/comparison on synthetic records; no services."""
import json
import time

import pytest

from arkb.agent.transports import OllamaClient
from evaluation.eval import (COSTS, METRICS, DeadlineExceeded, compare, deadline, load_questions, markdown,
                             score, summarize, write_json)


def test_question_file_is_valid_and_covers_every_type():
    questions = load_questions()
    assert len(questions) == 114
    assert {q['type'] for q in questions} == {'semantic_discovery', 'exploratory_retrieval', 'knowledge_qa', 'multi_hop_qa',
                                              'exact_lookup', 'direct_read', 'evidence_gap', 'no_retrieval', 'synthesis', 'browse', 'false_premise'}
    assert all(len(q['expected_sources']) >= 3 for q in questions if q['type'] == 'synthesis')
    assert all(q['expected_sources'] for q in questions if q['type'] in ('browse', 'false_premise'))
    assert all(q['expected_sources'] == [] for q in questions if q['type'] == 'no_retrieval')


def record(status='answered', *, cited=(), delivered=(), tools=('search',), cut=0, error=None):
    refs = {f'ev_{s}': {'source': s, 'content': 'body'} for s in delivered}
    events = [{'name': name, 'executed': True, 'delivered_to_conversation': True,
               'raw_result': {'results': [{'ref': f'ev_{s}'} for s in delivered]}} for name in tools]
    models = [{'response': {'done_reason': 'length'}}] * cut + [{'response': {'done_reason': 'stop'}}]
    return {'elapsed_ms': 2000, 'error': error, 'result': None if error else {
        'final': {'status': status, 'answer': 'x', 'citations': [{'source': s} for s in cited]},
        'observation': {'tools': events, 'models': models, 'evidence_references': refs, 'evidence': {'delivered_tokens': 30},
                        'usage': {'prompt_eval_count': {'known_total': 100}, 'eval_count': {'known_total': 10}}}}}


def test_scores_follow_the_question_type():
    q = {'id': 'q', 'type': 'knowledge_qa', 'expected_sources': ['a.md', 'b.md']}
    s = score(record(cited=['a.md', 'z.md'], delivered=['a.md', 'b.md', 'z.md']), q)
    assert s['answered'] and s['source_recall'] == 0.5 and s['source_precision'] == 0.5 and s['delivered_recall'] == 1.0
    assert s['complete'] is None and s['gap_respected'] is None and s['no_retrieval'] is None and s['read_only'] is None
    assert s['elapsed_s'] == 2.0 and s['model_requests'] == 1 and s['tool_calls'] == 1 and s['prompt_tokens'] == 100
    exact = score(record(cited=['a.md', 'b.md'], delivered=['a.md', 'b.md']), {**q, 'type': 'exact_lookup'})
    assert exact['complete'] is True
    assert score(record(cited=['a.md']), {**q, 'type': 'exact_lookup'})['complete'] is False
    assert score(record(cited=[]), {'id': 'e', 'type': 'exact_lookup', 'expected_sources': []})['complete'] is True
    gap = score(record('insufficient_evidence'), {'id': 'g', 'type': 'evidence_gap', 'expected_sources': []})
    assert gap['gap_respected'] is True and gap['answered'] is False and gap['source_recall'] is None
    partial = {'id': 'p', 'type': 'evidence_gap', 'expected_sources': ['a.md']}
    assert score(record('partial', cited=['a.md']), partial)['gap_respected'] is True
    assert score(record('answered', cited=['a.md']), partial)['gap_respected'] is False
    trap = {'id': 't', 'type': 'false_premise', 'expected_sources': ['a.md']}
    assert score(record('partial', cited=['a.md']), trap)['premise_flagged'] is True and score(record('answered', cited=['a.md']), trap)['premise_flagged'] is False
    assert score(record('answered'), q)['premise_flagged'] is None
    assert score(record(tools=()), {'id': 'n', 'type': 'no_retrieval', 'expected_sources': []})['no_retrieval'] is True
    assert score(record(tools=('search',)), {'id': 'n', 'type': 'no_retrieval', 'expected_sources': []})['no_retrieval'] is False
    assert score(record(tools=('read',)), {'id': 'd', 'type': 'direct_read', 'expected_sources': ['a.md']})['read_only'] is True
    failed = score(record(error={'type': 'X', 'message': 'y'}), q)
    assert failed['status'] == 'error' and failed['answered'] is False and failed['source_recall'] == 0.0
    assert score(record(cut=2), q)['responses_cut'] == 2


def rows(values, kind='knowledge_qa'):
    out = []
    for i, cited in enumerate(values):
        q = {'id': f'q{i}', 'type': kind, 'expected_sources': ['a.md']}
        row = {**record(cited=cited, delivered=['a.md']), 'question': q}
        row['scores'] = score(row, q)
        out.append(row)
    return out


def test_summary_markdown_and_compare(tmp_path):
    a, b = rows([[], ['a.md'], []]), rows([['a.md'], ['a.md'], []])
    summary = summarize(a)
    assert summary['all']['n'] == 3 and summary['all']['source_recall'] == pytest.approx(1 / 3)
    assert set(METRICS + COSTS) <= set(summary['all']) and 'knowledge_qa' in summary['by_type']
    text = markdown(summary, {'label': 'x', 'model': 'm', 'think': False, 'git_head': 'abcdef0123', 'dirty': False, 'questions': 3})
    assert '| all | 3 | 0 |' in text
    for name, group in (('a', a), ('b', b)):
        d = tmp_path / name
        d.mkdir()
        (d / 'results.jsonl').write_text('\n'.join(json.dumps(r) for r in group) + '\n')
        write_json(d / 'run.json', {'label': name, 'model': 'm', 'think': False, 'git_head': 'abcdef0123', 'dirty': False})
    table, text = compare(tmp_path / 'a', tmp_path / 'b')
    cell = table[('all', 'source_recall')]
    assert cell == {'a': pytest.approx(1 / 3), 'b': pytest.approx(2 / 3), 'diff': pytest.approx(1 / 3), 'wins': 1, 'ties': 2, 'losses': 0, 'n': 3}
    assert '| all | source_recall | 3 |' in text and 'B minus A' in text


class FakeHttp:
    def __init__(self):
        self.sent = []

    def post(self, path, *, json):
        self.sent.append(json)
        body = {'model': json['model'], 'done': True, 'done_reason': 'stop', 'message': {'role': 'assistant', 'content': '{}'}}
        return type('R', (), {'raise_for_status': lambda self: self, 'json': lambda self: body})()


def test_transport_keeps_the_callers_think_flag_and_rejects_another_model():
    client = OllamaClient.__new__(OllamaClient)
    client.model, client.options, client.think, client.http = 'qwen3.5:9b', {'num_ctx': 32768, 'num_predict': 4096}, True, FakeHttp()
    client.chat(model='qwen3.5:9b', messages=[], think=False, options={'temperature': 0})
    client.chat(model='qwen3.5:9b', messages=[])
    assert [r['think'] for r in client.http.sent] == [False, True]
    assert client.http.sent[0]['options'] == {'temperature': 0, 'num_ctx': 32768, 'num_predict': 4096}
    assert all(r['truncate'] is False and r['shift'] is False for r in client.http.sent)
    with pytest.raises(ValueError, match='differs'):
        client.chat(model='qwen3.5:4b', messages=[])


def test_deadline_interrupts_and_restores_the_timer():
    with pytest.raises(DeadlineExceeded):
        with deadline(0.05):
            time.sleep(1)
    with deadline(1):
        pass

"""Runner pieces that need no services: slice selection, the transport's request shape, the deadline."""
import time

import pytest

from evaluation.common import DeadlineExceeded, deadline
from evaluation.run import select_scenarios
from evaluation.transport import OllamaClient


def scenarios():
    return [{'id': 'v1', 'slice': 'v2', 'scope': 'v2-pilot'}, {'id': 'v2x', 'slice': 'v2', 'scope': 'v2-pilot'},
            {'id': 'e1', 'slice': 'exact-v2', 'scope': 'v2-pilot'},
            {'id': 'r1', 'slice': 'recall-fiqa', 'scope': 'fiqa', 'optional': True},
            {'id': 'm1', 'slice': 'musique', 'scope': 'ctx', 'optional': True, 'corpus': '/nowhere/c', 'sqlite': '/nowhere/i.sqlite'}]


def test_slice_selection_core_all_and_missing_inputs(tmp_path):
    present = {'corpus': str(tmp_path), 'sqlite': str(tmp_path / 'x.sqlite')}
    (tmp_path / 'x.sqlite').write_text('')
    scopes = {'v2-pilot': present, 'fiqa': {'corpus': '/nowhere', 'sqlite': '/nowhere.sqlite'}}
    chosen, skipped = select_scenarios(scenarios(), scopes, ['core'], 0)
    assert [s['id'] for s in chosen] == ['v1', 'v2x', 'e1'] and skipped == {}
    chosen, skipped = select_scenarios(scenarios(), scopes, ['all'], 0)
    assert [s['id'] for s in chosen] == ['v1', 'v2x', 'e1'] and set(skipped) == {'recall-fiqa', 'musique'}
    scopes['fiqa'] = present
    chosen, skipped = select_scenarios(scenarios(), scopes, ['recall-fiqa', 'v2'], 1)
    assert [s['id'] for s in chosen] == ['v1', 'r1'] and skipped == {}


class FakeHttp:
    def __init__(self):
        self.sent = []

    def post(self, path, *, json):
        self.sent.append(json)
        body = {'model': json['model'], 'done': True, 'done_reason': 'stop', 'message': {'role': 'assistant', 'content': '{}'}}
        return type('R', (), {'raise_for_status': lambda self: None, 'json': lambda self: body})()


def test_transport_keeps_the_callers_think_flag_and_rejects_another_model():
    client = OllamaClient.__new__(OllamaClient)
    client.model, client.options, client.think, client.http = 'qwen3.5:9b', {'num_ctx': 32768, 'num_predict': 4096}, True, FakeHttp()
    client.chat(model='qwen3.5:9b', messages=[], think=False, options={'temperature': 0})
    client.chat(model='qwen3.5:9b', messages=[])
    sent = client.http.sent
    assert [r['think'] for r in sent] == [False, True]
    assert sent[0]['options'] == {'temperature': 0, 'num_ctx': 32768, 'num_predict': 4096}
    assert all(r['truncate'] is False and r['shift'] is False for r in sent)
    with pytest.raises(ValueError, match='differs'):
        client.chat(model='qwen3.5:4b', messages=[])


def test_deadline_interrupts_and_restores_the_timer():
    with pytest.raises(DeadlineExceeded):
        with deadline(0.05):
            time.sleep(1)
    with deadline(1):
        pass
    with pytest.raises(ValueError):
        with deadline(0):
            pass

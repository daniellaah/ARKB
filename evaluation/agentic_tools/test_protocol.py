"""Protocol, transport and evidence-integrity fixtures beyond tool happy paths."""

import pytest

from .contract import ARM_BY_ID, OPTIONS, THINK, controlled_agent, fixed_rag, evidence_sets
from .labels import parse_judge
from .selection import attempt_key
from .test_contract import Capabilities, Client, response, obs
from .transport import LocalChatClient


@pytest.mark.parametrize('text,expected', [('correct: yes', True), ('**correct:** no', False),
                                         ('**correct**: yes', True), ('incorrect: yes', None),
                                         ('correct: yes\ncorrect: no', None), ('not graded', None)])
def test_judge_ambiguity_is_pending(text, expected):
    parsed = parse_judge(text)
    assert parsed['correct'] is expected
    assert parsed['parse_error'] is (expected is None)


def test_proposed_answer_cannot_cite_withheld_refs():
    base = Capabilities(('x ' * 8001,))
    s = __import__('evaluation.agentic_tools.contract', fromlist=['RestrictedSession']).RestrictedSession(
        __import__('evaluation.agentic_tools.contract', fromlist=['RestrictedTools']).RestrictedTools(base, ARM_BY_ID['A-S']))
    raw = s.invoke('search', {'query': 'x'})
    ref = raw['results'][0]['ref']
    assert s.invoke('finish', {'answer': 'x', 'status': 'answered', 'evidence_refs': [ref]})['status'] == 'recoverable_error'


@pytest.mark.parametrize('bad', ['not JSON', '{"answer":"x","answer":"y","status":"answered","evidence_refs":[]}',
                                '{"answer":"x","status":"answered","evidence_refs":["fake"]}'])
def test_fixed_invalid_final_is_retained_as_failure(bad):
    r = fixed_rag('q', tools=Capabilities(), arm=ARM_BY_ID['F-S'], client=Client([response('draft'), response(bad)]), observer=obs())
    assert r.final.status == 'error' and len(r.observation['models']) == 2
    assert r.observation['models'][1]['response']['message']['content'] == bad


def test_inference_adapter_accepts_no_label_arguments():
    with pytest.raises(TypeError):
        controlled_agent('q', tools=Capabilities(), arm=ARM_BY_ID['A-S'], client=None, observer=obs(), gold_answer='secret')


def test_failed_model_request_keeps_submitted_evidence_and_actual_options():
    def fail(request):
        raise RuntimeError('transport fixture')
    r = controlled_agent('q', tools=Capabilities(), arm=ARM_BY_ID['A-S'],
                         client=Client([response(calls=[('search', {'query': 'q'})]), fail]), observer=obs())
    assert r.final.status == 'error'
    assert evidence_sets(r.observation)['submitted'] == ['0.md', '1.md']
    for m in r.observation['models']:
        assert m['request']['options'] == OPTIONS
        assert m['request']['truncate'] is False and m['request']['shift'] is False


def test_attempt_identity_changes_for_every_registered_dimension():
    r = {'dataset': 'ds', 'id': 'q', 'variant': 'v0', 'arm': 'A-All', 'repetition': 0}
    original = attempt_key('protocol1', r)
    assert original != attempt_key('protocol2', r)
    for k in r:
        changed = {**r, k: 1 if k == 'repetition' else r[k] + 'x'}
        assert original != attempt_key('protocol1', changed)


def test_direct_transport_rejects_missing_anti_truncation_fields_before_http():
    client = LocalChatClient({'fixture': True})
    try:
        with pytest.raises(ValueError, match='context contract'):
            client.chat(model='qwen3.5:4b', options=OPTIONS, think=THINK, messages=[])
    finally:
        client.close()


def test_direct_transport_rejects_a_different_thinking_mode_before_http():
    client = LocalChatClient({'fixture': True})
    try:
        with pytest.raises(ValueError, match='context contract'):
            client.chat(model='qwen3.5:4b', options=OPTIONS, think=not THINK,
                        truncate=False, shift=False, messages=[])
    finally:
        client.close()

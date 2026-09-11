from dataclasses import asdict
import json
import re
import pytest
from arkb.agent import AgentBudget, AgentObserver, run_agent
from tests.agent.helpers import ScriptedModel, reply, tool_call, complete


def observer(**limits):
    return AgentObserver(budget=AgentBudget(**limits), counter=len, counter_identity='test-character-counter')


def normalized(value):
    return re.sub(r'ev_[0-9a-f]{32}_', 'ev_run_', json.dumps(value, sort_keys=True))


def test_observe_without_limits_preserves_messages_and_trace(tools):
    steps = [reply(calls=[tool_call('read', source='a.md')]), complete('answer')]
    plain, measured = ScriptedModel(*steps), ScriptedModel(*steps)
    a = run_agent('q', client=plain, tools=tools, model='fake')
    b = run_agent('q', client=measured, tools=tools, model='fake', observer=observer())
    assert normalized(plain.requests) == normalized(measured.requests)
    assert normalized(asdict(a.trace)) == normalized(asdict(b.trace))
    assert b.observation['tools'][0]['submitted_to_model']
    assert b.observation['usage']['eval_count']['total'] is None
    json.dumps(asdict(b), allow_nan=False)


def test_partial_batch_limit_records_skips_and_finalization_submission(tools):
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md')]*3), complete('answer', constrained=True))
    result = run_agent('q', client=model, tools=tools, model='fake', observer=observer(max_tool_calls=1))
    assert result.stop_reason == 'final' and len(model.requests) == 2
    assert [e['status'] for e in result.observation['tools']] == ['success', 'skipped', 'skipped']
    assert result.observation['tools'][0]['submitted_to_model']
    assert not any(e['executed'] for e in result.observation['tools'][1:])
    assert all(c.result['error']['code'] == 'budget_exhausted' for c in result.trace.tool_calls[1:])


def test_query_limit_does_not_spend_read_allowance(tools, engine):
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md'), tool_call('search', query='q')]),
                          complete('answer', constrained=True))
    result = run_agent('q', client=model, tools=tools, model='fake', observer=observer(max_query_calls=0,max_read_calls=1))
    engine.search.assert_not_called()
    assert result.observation['budget_stop_reason'] == 'max_query_calls'
    assert result.final.citations[0]['source'] == 'a.md'


def test_evidence_limit_counts_repeats_and_retains_but_withholds_oversized_result(tools, documents):
    size = len(documents.read(source='a.md').content)
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md')]*2), complete('answer', constrained=True))
    result = run_agent('q', client=model, tools=tools, model='fake', observer=observer(max_evidence_tokens=size))
    assert result.observation['evidence'] == {'returned_tokens':size*2,'delivered_tokens':size,'unique_exact_excerpt_tokens':size}
    last = result.observation['tools'][-1]
    assert last['executed'] and last['raw_result']['result']['content'] == documents.read(source='a.md').content
    assert not last['delivered_to_conversation']
    assert result.trace.tool_calls[-1].result['error']['code'] == 'budget_exhausted'


def test_recoverable_error_and_fatal_error_are_distinct(tools, engine):
    engine.search.side_effect = RuntimeError('backend corrupt')
    model = ScriptedModel(reply(calls=[tool_call('read', source='missing.md'),tool_call('search', query='q'),tool_call('read',source='a.md')]))
    result = run_agent('q', client=model, tools=tools, model='fake', observer=observer())
    assert [e['status'] for e in result.observation['tools']] == ['recoverable_error','fatal_error','skipped']
    assert result.observation['error']['stage'] == 'tool_execution'
    assert result.trace.tool_calls[-1].result is None


def test_native_usage_keeps_missing_separate_from_zero_and_records_protocol_failure(tools):
    first = reply(calls=[tool_call('read', source='a.md')]); first.prompt_eval_count=10; first.eval_count=2
    last = reply('partial',done_reason='length'); last.prompt_eval_count=0; last.eval_count=3
    result = run_agent('q',client=ScriptedModel(first,last),tools=tools,model='fake',observer=observer())
    report = result.observation
    assert report['usage']['prompt_eval_count']['total'] == 10 and report['usage']['eval_count']['total'] == 5
    assert report['error']['stage'] == 'model_protocol'
    assert report['models'][-1]['response']['message']['content'] == 'partial'


@pytest.mark.parametrize('during', ['model','tool'])
def test_deadline_retains_late_raw_result_but_never_delivers_it(tools, monkeypatch, during):
    clock = [0.]
    original_read = tools.read
    def late_model(messages):
        clock[0] = 1.
        return reply('late answer')
    def late_tool(**args):
        result = original_read(**args)
        clock[0] = 1.
        return result
    if during == 'tool': monkeypatch.setattr(tools,'read',late_tool)
    model = ScriptedModel(late_model if during == 'model' else reply(calls=[tool_call('read',source='a.md')]))
    o = AgentObserver(budget=AgentBudget(max_elapsed_ms=100),counter=len,counter_identity='chars',clock=lambda:clock[0])
    result = run_agent('q',client=model,tools=tools,model='fake',observer=o)
    assert result.stop_reason == 'budget' and result.response is None and result.final.status == 'error'
    assert result.observation['elapsed_ms'] == 1000
    assert not any(e['delivered_to_conversation'] for e in result.observation['tools'])


def test_observer_reuse_and_missing_counter_are_rejected(tools):
    with pytest.raises(ValueError,match='counter'): AgentObserver(budget=AgentBudget(max_evidence_tokens=1))
    with pytest.raises(ValueError): AgentBudget(max_query_calls=True)
    o=observer(); run_agent('q',client=ScriptedModel(complete('ok')),tools=tools,model='fake',observer=o)
    with pytest.raises(ValueError,match='fresh'):
        run_agent('q',client=ScriptedModel(complete('ok')),tools=tools,model='fake',observer=o)

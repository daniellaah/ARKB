import json
from dataclasses import asdict
import pytest
from arkb.agent import AgentBudget, AgentObserver, run_agent
from tests.agent.helpers import ScriptedModel, complete, reply, tool_call


def finish(messages):
    hits = [json.loads(m['content'])['result'] for m in messages if m['role'] == 'tool'
            and 'result' in json.loads(m['content'])]
    return reply(calls=[tool_call('finish', answer='éø café', status='answered',
                                  evidence_refs=[hits[0]['ref']] if hits else [])])


def final_json(messages):
    hits = [json.loads(m['content'])['result'] for m in messages if m['role'] == 'tool'
            and 'result' in json.loads(m['content'])]
    return reply(json.dumps({'answer': 'Answer with ```json and "quotes"\ninside text.',
        'status': 'answered' if hits else 'insufficient_evidence',
        'evidence_refs': [hits[0]['ref']] if hits else []}))


def test_invalid_then_corrected_call_then_canonical_finish(tools):
    model = ScriptedModel(reply(calls=[tool_call('read', document_id='a.md')]),
                          reply(calls=[tool_call('read', source='a.md')]), finish)
    result = run_agent('Read a.md', client=model, tools=tools, model='fake')
    assert result.stop_reason == 'final'
    assert result.final.status == 'answered'
    assert result.final.citations[0]['source'] == 'a.md'
    assert [e['status'] for e in result.observation['tools']] == ['recoverable_error', 'success', 'success']
    assert result.trace.tool_calls[0].result['error']['code'] == 'invalid_arguments'
    assert result.observation['models'][1]['request']['messages'][-1]['role'] == 'tool'
    assert result.observation['evidence_references']
    assert result.observation['reference_resolutions']
    assert json.loads(result.final.to_json()) == asdict(result.final)


@pytest.mark.parametrize('limits,expected', [({'max_tool_calls': 1}, 'max_tool_calls'),
    ({'max_read_calls': 1}, 'max_read_calls'), ({'max_evidence_tokens': 42}, 'max_evidence_tokens')])
def test_budget_exhaustion_delivers_prior_evidence_to_reserved_finalization(tools, limits, expected):
    # a.md delivers 39 characters; with 42 the remaining allowance is below one tenth, so the run closes.
    observer = AgentObserver(budget=AgentBudget(**limits), counter=len, counter_identity='chars')
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md')]*3), final_json)
    result = run_agent('Read a.md', client=model, tools=tools, model='fake', observer=observer)
    assert result.stop_reason == 'final' and result.final.termination_reason == expected
    assert len(model.requests) == 2 and 'tools' not in model.requests[-1]
    assert model.requests[-1]['format']['type'] == 'object'
    assert result.observation['tools'][0]['submitted_to_model']
    assert not result.observation['tools'][-1]['executed']
    assert result.final.citations[0]['source'] == 'a.md'


def test_max_turns_reserves_final_request_without_more_tool_work(tools):
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md')]), final_json)
    result = run_agent('Read a.md', client=model, tools=tools, model='fake', max_turns=2)
    assert result.state.turn == 2 and result.stop_reason == 'final'
    assert result.final.termination_reason == 'max_turns'
    assert 'tools' not in model.requests[-1]
    assert json.loads(result.final.to_json())['answer'].startswith('Answer with ```json')


def test_one_turn_is_a_finalization_opportunity(tools):
    model = ScriptedModel(final_json)
    result = run_agent('Read a.md', client=model, tools=tools, model='fake', max_turns=1)
    assert result.final.status == 'insufficient_evidence'
    assert 'tools' not in model.requests[0]
    assert result.state.turn == 1


def test_invalid_finish_reference_can_be_corrected(tools):
    model = ScriptedModel(reply(calls=[tool_call('finish', answer='x', status='answered', evidence_refs=['bad'])]),
                          reply(calls=[tool_call('read', source='a.md')]), finish)
    result = run_agent('Read a.md', client=model, tools=tools, model='fake')
    assert result.stop_reason == 'final'
    assert result.trace.tool_calls[0].result['status'] == 'recoverable_error'


def test_invalid_final_json_is_a_structured_failure_not_prose_extraction(tools):
    result = run_agent('q', client=ScriptedModel(reply('```json\n{}\n```')), tools=tools, model='fake', max_turns=1)
    assert result.stop_reason == 'error' and result.final.status == 'error'
    assert result.final.termination_reason == 'invalid_final_output'
    assert result.response is None and result.final.error


def test_fatal_backend_returns_visible_structured_failure(tools, engine):
    engine.search.side_effect = RuntimeError('snapshot corrupt')
    result = run_agent('q', client=ScriptedModel(reply(calls=[tool_call('search', query='x')])), tools=tools, model='fake')
    assert result.stop_reason == 'error' and result.final.status == 'error'
    assert result.observation['tools'][0]['status'] == 'fatal_error'
    assert result.trace.tool_calls[0].result['status'] == 'fatal_error'
    assert result.final.error['type'] == 'RuntimeError'


def test_invalid_calls_consume_budget(tools):
    model = ScriptedModel(reply(calls=[tool_call('no_such_tool')]*3), final_json)
    result = run_agent('q', client=model, tools=tools, model='fake',
        observer=AgentObserver(budget=AgentBudget(max_tool_calls=1)))
    assert result.final.termination_reason == 'max_tool_calls'
    assert sum(e['executed'] for e in result.observation['tools']) == 1
    assert result.observation['tools'][0]['status'] == 'recoverable_error'


def test_duplicate_final_fields_are_not_silently_overwritten(tools):
    text = '{"answer":"one","answer":"two","status":"answered","evidence_refs":[]}'
    result = run_agent('Hello', client=ScriptedModel(reply(text)), tools=tools, model='fake', max_turns=1)
    assert result.final.status == 'error' and result.final.termination_reason == 'invalid_final_output'


def test_many_invalid_finish_calls_consume_one_turn_and_cannot_loop(tools):
    calls = [tool_call('finish', answer='x', status='answered', evidence_refs=['invalid'])]*30
    model = ScriptedModel(reply(calls=calls), final_json)
    result = run_agent('q', client=model, tools=tools, model='fake', max_turns=2,
        observer=AgentObserver(budget=AgentBudget(max_tool_calls=0)))
    assert result.stop_reason == 'final' and result.state.turn == 2
    assert sum(e['executed'] for e in result.observation['tools']) == 1
    assert all(e['skip_reason'] == 'invalid_finish' for e in result.observation['tools'][1:])


def test_oversized_result_is_withheld_but_collection_continues_while_allowance_remains(tools):
    observer = AgentObserver(budget=AgentBudget(max_evidence_tokens=60), counter=len, counter_identity='chars')
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md'), tool_call('read', source='a.md')]),
                          reply(calls=[tool_call('read', source='empty.md')]),
                          complete('answer', constrained=True), complete('answer', constrained=True))
    result = run_agent('Read', client=model, tools=tools, model='fake', observer=observer)
    events = result.observation['tools']
    assert [e['delivered_to_conversation'] for e in events[:3]] == [True, False, True]
    assert result.trace.tool_calls[1].result['error']['code'] == 'evidence_too_large'
    assert 'expand=section' in result.trace.tool_calls[1].result['error']['message']
    assert result.observation['budget_stop_reason'] is None
    assert result.stop_reason == 'final' and result.final.termination_reason == 'final_requested'
    assert {c['source'] for c in result.final.citations} == {'a.md', 'empty.md'}

"""Detached diagnostics over real document tools and a scripted provider."""
from dataclasses import asdict
import json
from unittest.mock import Mock
from ollama import ResponseError
import pytest
from arkb.agent import AgentTrace, run_agent
from arkb.runtime import Runtime
from tests.agent.helpers import ScriptedModel, reply, tool_call, complete


def test_final_trace_records_query_turns_order_arguments_results_and_exact_response(tools):
    calls = [tool_call('read', source='a.md'), tool_call('read', source='b.md')]
    client = ScriptedModel(reply('Reading', calls=calls), reply(calls=[calls[0]]), complete('  Answer\n'))
    with Runtime() as runtime:
        result = runtime.run_agent('  Find notes\n', client=client, tools=tools, model='fake')
    trace = result.trace
    assert isinstance(trace, AgentTrace)
    assert (trace.query, trace.turns, trace.final_response, trace.stop_reason) == ('  Find notes\n',3,'  Answer\n','final')
    assert [(c.turn,c.name,c.arguments) for c in trace.tool_calls[:-1]] == [
        (turn,'read',{'source':source}) for turn,source in [(1,'a.md'),(1,'b.md'),(2,'a.md')]]
    assert all(c.result['status']=='success' for c in trace.tool_calls)
    assert json.loads(json.dumps(asdict(trace),ensure_ascii=False)) == asdict(trace)
    assert set(asdict(result)) == {'response','stop_reason','state','final','observation'}
    assert set(asdict(result.state)) == {'messages','turn'}
    trace.tool_calls[0].arguments['source']='changed.md'
    trace.tool_calls[0].result['result']['content']='changed'
    assert result.trace.tool_calls[0].arguments == {'source':'a.md'}
    assert result.trace.tool_calls[0].result['result']['content'] != 'changed'
    ref = result.trace.tool_calls[0].result['result']['ref']
    assert result.observation['evidence_references'][ref]['document_id']


def test_direct_constrained_final_has_no_tool_trajectory(tools):
    client = ScriptedModel(complete('Hello',status='answered',constrained=True))
    result = run_agent('Hello',client=client,tools=tools,model='fake',max_turns=1)
    assert asdict(result.trace) == {'query':'Hello','turns':1,'tool_calls':[], 'final_response':'Hello','stop_reason':'final'}


@pytest.mark.parametrize('max_turns',[1,3])
def test_final_turn_tool_requests_are_retained_but_not_executed(tools,max_turns):
    calls=[tool_call('read',source='a.md'),tool_call('read',source='b.md')]
    client=Mock(chat=Mock(return_value=reply('Still reading',calls=calls)))
    result=run_agent('Read notes',client=client,tools=tools,model='fake',max_turns=max_turns)
    assert result.final.termination_reason == 'invalid_final_output'
    assert result.state.turn == max_turns and result.response is None
    assert len(result.trace.tool_calls) == 2*max_turns
    assert all(c.result and c.result['status']=='success' for c in result.trace.tool_calls[:-2])
    assert all(c.result is None for c in result.trace.tool_calls[-2:])
    assert sum(e['executed'] for e in result.observation['tools']) == 2*(max_turns-1)


@pytest.mark.parametrize('completed_turns',[0,1])
@pytest.mark.parametrize('error_type',[ConnectionError,ResponseError])
def test_model_error_preserves_completed_trajectory_and_counts_failed_request(tools,completed_turns,error_type):
    error=error_type('model offline')
    client=Mock(chat=Mock(side_effect=[reply(calls=[tool_call('read',source='a.md')])]*completed_turns+[error]))
    with Runtime() as runtime:
        result=runtime.run_agent('Read notes',client=client,tools=tools,model='fake')
    assert (result.trace.turns,result.trace.stop_reason,result.trace.final_response)==(completed_turns+1,'error',None)
    assert len(result.trace.tool_calls)==completed_turns
    if completed_turns:
        assert result.trace.tool_calls[0].result['result']['source']=='a.md'
    assert result.final.error['type']==error_type.__name__
    assert client.chat.call_count==completed_turns+1


def test_fatal_error_retains_prior_observations_and_partial_batch(tools,engine):
    engine.search.side_effect=RuntimeError('corrupt index')
    first=tool_call('read',source='a.md')
    batch=[tool_call('match',query='needle'),tool_call('search',query='q'),tool_call('read',source='b.md')]
    model=ScriptedModel(reply(calls=[first]),reply('Inspecting',calls=batch))
    result=run_agent('Read notes',client=model,tools=tools,model='fake')
    assert result.trace.turns==2 and result.stop_reason=='error'
    assert [c.result['status'] if c.result else None for c in result.trace.tool_calls]==['success','success','fatal_error',None]
    assert result.observation['tools'][-1]['status']=='skipped'
    assert result.observation['tools'][-1]['executed'] is False


@pytest.mark.parametrize('response',[reply('partial',done_reason='length'),reply(calls=[tool_call('read',source='b.md')],done_reason='length')])
def test_protocol_errors_keep_prior_observations_without_claiming_a_final(tools,response):
    result=run_agent('Read notes',client=ScriptedModel(reply(calls=[tool_call('read',source='a.md')]),response),tools=tools,model='fake')
    assert result.trace.turns==2 and result.stop_reason=='error' and result.response is None
    assert result.trace.tool_calls[0].result['result']['source']=='a.md'
    assert result.observation['models'][-1]['response']['done_reason']=='length'


def test_exception_that_rejects_attributes_still_has_structured_diagnostics(tools):
    class ReadOnlyError(RuntimeError):
        def __setattr__(self,name,value):raise TypeError('read-only')
    result=run_agent('q',client=Mock(chat=Mock(side_effect=ReadOnlyError('model failed'))),tools=tools,model='fake')
    assert result.final.error['type']=='ReadOnlyError' and result.final.error['message']=='model failed'

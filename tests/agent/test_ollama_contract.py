"""Offline wire-contract checks using the real SDK and an in-process transport."""

import json
from unittest.mock import Mock

import httpx
from ollama import Client
import pytest

from arkb.agent import AgentTools, run_agent
from arkb.retrieval import ExactRetriever, RetrievalEngine
from tests.agent.helpers import tool_call, complete


@pytest.mark.parametrize('think', [True, False])
def test_ollama_serializes_tool_calls_and_all_observations(tools, think):
    requests = []
    calls = [tool_call('read', source='a.md'), tool_call('read', source='b.md')]

    def handle(request):
        assert request.url.path == '/api/chat'
        body = json.loads(request.content)
        assert body['think'] is think
        requests.append(body)
        definitions = {t['function']['name']: t['function'] for t in body['tools']}
        assert set(definitions) == {'match', 'search', 'list', 'read', 'finish'}
        assert set(definitions['search']['parameters']['properties']) == {'query', 'source', 'limit', 'mode'}
        assert {'ref', 'source'} <= definitions['read']['parameters']['properties'].keys()
        if len(requests) == 1:
            message = {'role': 'assistant', 'tool_calls': calls}
        else:
            assert body['messages'][2]['tool_calls'] == calls
            observations = body['messages'][3:]
            assert [m['role'] for m in observations] == ['tool', 'tool']
            assert [m['tool_name'] for m in observations] == ['read', 'read']
            assert [json.loads(m['content'])['result']['source'] for m in observations] == ['a.md', 'b.md']
            message = complete('finished')(body['messages']).message.model_dump(exclude_none=True)
        return httpx.Response(200, json={'message': message, 'done': True, 'done_reason': 'stop'})

    with Client(host='http://ollama.test', transport=httpx.MockTransport(handle), trust_env=False) as client:
        result = run_agent('Read a.md and b.md', client=client, tools=tools, model='fake', think=think)
    assert result.stop_reason == 'final'
    assert result.state.turn == len(requests) == 2
    assert result.state.tool_calls[:-1] == calls


@pytest.mark.parametrize('modes', [('bm25',), ('semantic',), ('bm25', 'semantic')])
def test_model_instructions_and_schema_only_advertise_available_modes(documents, modes):
    engine = RetrievalEngine(**{mode: Mock() for mode in modes})
    tools = AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine, mode=modes[0])
    available = set(modes) | ({'hybrid'} if len(modes) == 2 else set())

    def handle(request):
        body = json.loads(request.content)
        search = next(t['function'] for t in body['tools'] if t['function']['name'] == 'search')
        parameter = search['parameters']['properties']['mode']
        assert set(parameter['enum']) == available | {None}
        instructions = ' '.join([body['messages'][0]['content'], search['description'], parameter['description']])
        for unsupported in {'bm25', 'semantic', 'hybrid'} - available:
            assert unsupported not in instructions.split(), instructions
        return httpx.Response(200, json={'message': {'role': 'assistant', 'tool_calls': [tool_call('finish', answer='finished', status='answered', evidence_refs=[])]}, 'done': True})

    with Client(host='http://ollama.test', transport=httpx.MockTransport(handle), trust_env=False) as client:
        assert run_agent('Hello', client=client, tools=tools, model='fake').response == 'finished'


@pytest.mark.parametrize('arguments', ['{"source":', 'not json', [], None])
def test_malformed_tool_arguments_recover_at_sdk_boundary(tools, arguments):
    requests = []
    def handle(request):
        body = json.loads(request.content); requests.append(body)
        if len(requests) == 1:
            message = {'role': 'assistant', 'tool_calls': [{'function': {'name': 'read', 'arguments': arguments}}]}
        elif len(requests) == 2:
            message = {'role': 'assistant', 'tool_calls': [tool_call('read', source='a.md')]}
        else:
            message = complete('finished')(body['messages']).message.model_dump(exclude_none=True)
        return httpx.Response(200, json={'message': message, 'done': True})
    with Client(host='http://ollama.test', transport=httpx.MockTransport(handle), trust_env=False) as client:
        result = run_agent('Read a.md', client=client, tools=tools, model='fake')
    assert result.stop_reason == 'final' and result.state.turn == 3
    assert result.observation['models'][0]['status'] == 'recoverable_error'
    assert result.observation['models'][0]['error']['validation'][0]['input'] == arguments
    assert result.observation['tools'][0]['status'] == 'recoverable_error'
    assert [c.turn for c in result.trace.tool_calls] == [1, 2, 3]

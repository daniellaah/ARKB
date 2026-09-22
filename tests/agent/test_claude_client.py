"""The Claude transport behind the loop's Ollama-shaped contract; a scripted Messages API, no network."""
import json
from types import SimpleNamespace

import pytest

from arkb.agent import run_agent
from arkb.agent.claude_client import ClaudeClient, convert_tools, output_schema
from arkb.agent.tools import FINAL_SCHEMA


class ScriptedMessages:
    """Answers messages.create with scripted content blocks and records every request."""

    def __init__(self, *turns):
        self.turns, self.requests = list(turns), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        blocks, stop = self.turns.pop(0)
        return SimpleNamespace(content=blocks, stop_reason=stop, model='claude-opus-5', stop_details=None,
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0))


def scripted(*turns):
    api = SimpleNamespace(messages=ScriptedMessages(*turns))
    return ClaudeClient('claude-opus-5', client=api), api.messages


def thinking(text='reasoning', signature='sig'):
    return {'type': 'thinking', 'thinking': text, 'signature': signature}


def tool_use(name, tool_id, **arguments):
    return {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': arguments}


def test_request_translation_and_effort_follow_the_think_flag():
    client, api = scripted(([thinking(), {'type': 'text', 'text': 'hello'}], 'end_turn'))
    tools = [{'type': 'function', 'function': {'name': 'search', 'description': 'd', 'parameters': {'type': 'object', 'properties': {}}}}]
    response = client.chat(model='ignored', messages=[{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'q'}],
                           think=True, tools=tools, options={'temperature': 0}, stream=False)
    request = api.requests[0]
    assert request['system'] == 'S' and request['messages'] == [{'role': 'user', 'content': [{'type': 'text', 'text': 'q'}]}]
    assert request['thinking'] == {'type': 'adaptive'} and request['output_config'] == {'effort': 'high'}
    assert request['tools'] == convert_tools(tools) and 'temperature' not in request and request['model'] == 'claude-opus-5'
    assert response.message.content == 'hello' and response.message.thinking == 'reasoning' and response.done_reason == 'stop'
    assert response.prompt_eval_count == 100 and response.eval_count == 20
    client, api = scripted(([{'type': 'text', 'text': '{}'}], 'end_turn'))
    client.chat(messages=[{'role': 'user', 'content': 'q'}], think=False, format=FINAL_SCHEMA)
    request = api.requests[0]
    assert request['output_config'] == {'effort': 'low', 'format': {'type': 'json_schema', 'schema': output_schema(FINAL_SCHEMA)}}
    assert 'uniqueItems' not in json.dumps(request['output_config']) and 'tools' not in request


def test_assistant_turns_are_replayed_with_their_blocks_and_tool_results_follow_their_ids():
    client, api = scripted(([thinking(), tool_use('search', 'toolu_1', query='a'), tool_use('read', 'toolu_2', source='x.md')], 'tool_use'),
                           ([tool_use('search', 'toolu_3', query='a')], 'tool_use'),
                           ([{'type': 'text', 'text': 'done'}], 'end_turn'))
    messages = [{'role': 'user', 'content': 'q'}]
    first = client.chat(messages=messages, think=True).message.model_dump(exclude_none=True)
    assert [c['function']['name'] for c in first['tool_calls']] == ['search', 'read']
    # The loop strips thinking before replay and appends one tool message per call.
    replay = [{k: v for k, v in first.items() if k != 'thinking'}]
    messages += replay + [{'role': 'tool', 'content': '{"r": 1}', 'tool_name': 'search'},
                          {'role': 'tool', 'content': '{"r": 2}', 'tool_name': 'read'}]
    second = client.chat(messages=messages, think=True).message.model_dump(exclude_none=True)
    sent = api.requests[1]['messages']
    assert sent[1]['content'][0] == thinking() and [b['id'] for b in sent[1]['content'] if b['type'] == 'tool_use'] == ['toolu_1', 'toolu_2']
    assert sent[2] == {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': '{"r": 1}'},
                                                   {'type': 'tool_result', 'tool_use_id': 'toolu_2', 'content': '{"r": 2}'}]}
    # An identical repeated turn keeps its own ids, consumed in order.
    messages += [{k: v for k, v in second.items() if k != 'thinking'}, {'role': 'tool', 'content': '{"r": 3}', 'tool_name': 'search'},
                 {'role': 'system', 'content': 'Evidence collection is closed.'}]
    client.chat(messages=messages, think=False, format=FINAL_SCHEMA)
    sent = api.requests[2]['messages']
    assert [b['id'] for b in sent[3]['content'] if b['type'] == 'tool_use'] == ['toolu_3']
    assert sent[4]['content'][0]['tool_use_id'] == 'toolu_3'
    assert sent[5] == {'role': 'user', 'content': [{'type': 'text', 'text': 'Evidence collection is closed.'}]}


def test_unknown_turns_are_rebuilt_and_termination_is_mapped():
    client, api = scripted(([{'type': 'text', 'text': 'cut'}], 'max_tokens'), ([{'type': 'text', 'text': 'no'}], 'refusal'))
    messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'earlier', 'tool_calls': [{'function': {'name': 'list', 'arguments': {}}}]},
                {'role': 'tool', 'content': '{}', 'tool_name': 'list'}]
    response = client.chat(messages=messages, think=True)
    assert response.done_reason == 'length'
    rebuilt = api.requests[0]['messages'][1]['content']
    assert rebuilt[0] == {'type': 'text', 'text': 'earlier'} and rebuilt[1]['type'] == 'tool_use' and rebuilt[1]['name'] == 'list'
    assert api.requests[0]['messages'][2]['content'][0]['tool_use_id'] == rebuilt[1]['id']
    with pytest.raises(ValueError, match='declined'):
        client.chat(messages=[{'role': 'user', 'content': 'q'}], think=True)
    with pytest.raises(ValueError, match='no preceding tool call'):
        client.convert_messages([{'role': 'user', 'content': 'q'}, {'role': 'tool', 'content': '{}', 'tool_name': 'x'}])
    with pytest.raises(ValueError, match='effort'):
        ClaudeClient(client=SimpleNamespace(), effort='enormous')


def test_the_product_loop_runs_end_to_end_through_the_claude_transport(tools):
    def finish_from(messages):
        refs = []
        for m in messages:
            if m['role'] == 'user':
                for block in m['content']:
                    if block.get('type') == 'tool_result':
                        for hit in json.loads(block['content']).get('results', []):
                            refs.append(hit['ref'])
        return [tool_use('finish', 'toolu_fin', answer='beta', status='answered', evidence_refs=refs[:1])]
    api = SimpleNamespace(messages=None)
    turns = [([thinking(), tool_use('match', 'toolu_m', query='foo')], 'tool_use'), None]
    scripted_api = ScriptedMessages(*turns)

    def create(**kwargs):
        if scripted_api.turns[0] is None:
            scripted_api.turns[0] = (finish_from(kwargs['messages']), 'tool_use')
        return scripted_api.create(**kwargs)
    api.messages = SimpleNamespace(create=create, requests=scripted_api.requests)
    client = ClaudeClient('claude-opus-5', client=api)
    result = run_agent('which notes mention foo?', client=client, tools=tools, model='claude-opus-5', think=True)
    assert result.final.status == 'answered' and result.final.answer == 'beta' and len(result.final.citations) == 1
    assert [len(r['messages']) for r in scripted_api.requests] == [1, 3]
    assert scripted_api.requests[1]['messages'][1]['content'][0] == thinking()
    assert len(client.usage) == 2

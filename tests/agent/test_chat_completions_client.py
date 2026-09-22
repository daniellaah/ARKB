"""The chat-completions transport (DeepSeek shape) behind the loop's contract; a scripted API, no network."""
import json

import pytest

from arkb.agent import run_agent
from arkb.agent.chat_completions_client import ChatCompletionsClient
from arkb.agent.tools import FINAL_SCHEMA


class FakeHttp:
    def __init__(self, *bodies):
        self.bodies, self.requests = list(bodies), []

    def post(self, path, *, json):
        self.requests.append(json)
        body = self.bodies.pop(0)
        status = 200 if 'choices' in body else 400
        return type('R', (), {'status_code': status, 'text': str(body), 'json': lambda self: body})()

    def close(self):
        pass


def completion(content='', calls=(), finish='stop', reasoning=None):
    message = {'role': 'assistant', 'content': content}
    if reasoning:
        message['reasoning_content'] = reasoning
    if calls:
        message['tool_calls'] = [{'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}} for cid, name, args in calls]
    return {'model': 'deepseek-reasoner', 'choices': [{'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 50, 'completion_tokens': 7, 'completion_tokens_details': {'reasoning_tokens': 3}}}


def client_with(*bodies):
    http = FakeHttp(*bodies)
    return ChatCompletionsClient('deepseek-reasoner', http=http), http


def test_request_shape_tools_and_json_mode():
    client, http = client_with(completion('hello', reasoning='why'))
    tools = [{'type': 'function', 'function': {'name': 'search', 'description': 'd', 'parameters': {'type': 'object', 'properties': {}}}}]
    response = client.chat(model='ignored', messages=[{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'q'}],
                           think=True, tools=tools, options={'temperature': 0}, stream=False)
    body = http.requests[0]
    assert body['model'] == 'deepseek-reasoner' and body['messages'] == [{'role': 'system', 'content': 'S'}, {'role': 'user', 'content': 'q'}]
    assert body['tools'] == tools and body['temperature'] == 0 and 'response_format' not in body
    assert response.message.content == 'hello' and response.message.thinking == 'why' and response.prompt_eval_count == 50
    client, http = client_with(completion('{}'))
    client.chat(messages=[{'role': 'user', 'content': 'q'}], format=FINAL_SCHEMA)
    body = http.requests[0]
    assert body['response_format'] == {'type': 'json_object'} and body['messages'][-1]['role'] == 'system'
    assert 'evidence_refs' in body['messages'][-1]['content'] and 'tools' not in body


def test_tool_call_ids_are_kept_per_turn_and_echoed_by_tool_results():
    client, http = client_with(completion(calls=[('call_a', 'search', {'query': 'x'}), ('call_b', 'read', {'source': 'x.md'})], finish='tool_calls'),
                               completion(calls=[('call_c', 'search', {'query': 'x'})], finish='tool_calls'),
                               completion('done'))
    messages = [{'role': 'user', 'content': 'q'}]
    first = client.chat(messages=messages).message.model_dump(exclude_none=True)
    assert first['tool_calls'][0]['function']['arguments'] == {'query': 'x'}
    messages += [{k: v for k, v in first.items() if k != 'thinking'},
                 {'role': 'tool', 'content': '{"r": 1}', 'tool_name': 'search'}, {'role': 'tool', 'content': '{"r": 2}', 'tool_name': 'read'}]
    second = client.chat(messages=messages).message.model_dump(exclude_none=True)
    sent = http.requests[1]['messages']
    assert [c['id'] for c in sent[1]['tool_calls']] == ['call_a', 'call_b'] and sent[1]['tool_calls'][0]['function']['arguments'] == '{"query": "x"}'
    assert sent[2] == {'role': 'tool', 'tool_call_id': 'call_a', 'content': '{"r": 1}'} and sent[3]['tool_call_id'] == 'call_b'
    assert 'reasoning_content' not in json.dumps(sent)
    messages += [{k: v for k, v in second.items() if k != 'thinking'}, {'role': 'tool', 'content': '{"r": 3}', 'tool_name': 'search'}]
    client.chat(messages=messages)
    sent = http.requests[2]['messages']
    assert sent[4]['tool_calls'][0]['id'] == 'call_c' and sent[5]['tool_call_id'] == 'call_c'


def test_unknown_turns_errors_and_length():
    client, http = client_with(completion('cut', finish='length'), {'error': {'message': 'bad'}})
    messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'list', 'arguments': {}}}]},
                {'role': 'tool', 'content': '{}', 'tool_name': 'list'}]
    assert client.chat(messages=messages).done_reason == 'length'
    rebuilt = http.requests[0]['messages'][1]
    assert rebuilt['tool_calls'][0]['function'] == {'name': 'list', 'arguments': '{}'} and http.requests[0]['messages'][2]['tool_call_id'] == rebuilt['tool_calls'][0]['id']
    with pytest.raises(ValueError, match='error 400'):
        client.chat(messages=[{'role': 'user', 'content': 'q'}])
    with pytest.raises(ValueError, match='No API key'):
        ChatCompletionsClient('deepseek-chat', api_key_env='ARKB_TEST_MISSING_KEY')


def test_the_product_loop_runs_end_to_end_through_chat_completions(tools):
    http = FakeHttp(completion(calls=[('call_m', 'match', {'query': 'foo'})], finish='tool_calls'))
    client = ChatCompletionsClient('deepseek-chat', http=http)
    original_post = http.post

    def post(path, *, json):
        if not http.bodies:
            refs = [hit['ref'] for m in json['messages'] if m['role'] == 'tool' for hit in __import__('json').loads(m['content']).get('results', [])]
            http.bodies.append(completion(calls=[('call_f', 'finish', {'answer': 'beta', 'status': 'answered', 'evidence_refs': refs[:1]})], finish='tool_calls'))
        return original_post(path, json=json)
    http.post = post
    result = run_agent('which notes mention foo?', client=client, tools=tools, model='deepseek-chat', think=False)
    assert result.final.status == 'answered' and result.final.answer == 'beta' and len(result.final.citations) == 1
    assert [len(r['messages']) for r in http.requests] == [2, 4] and http.requests[1]['messages'][3]['tool_call_id'] == 'call_m'

"""OpenAI-compatible chat completions (DeepSeek and similar) behind the loop's Ollama-shaped contract.

The loop's dialect is Ollama's: assistant messages carry `content`, `thinking`
and `tool_calls` with dict arguments; tool results are `role: tool` messages
without ids; `format` is a JSON schema for the reserved finalization. Chat
completions differ in three places, handled here: tool calls carry ids that
tool results must echo, arguments travel as JSON strings, and JSON output is
requested with `response_format` plus an instruction that names the schema.
Assistant turns are replayed with the ids the provider issued, kept per turn
and consumed in order; `reasoning_content` is never sent back, as DeepSeek
requires. Thinking is a property of the model name (deepseek-reasoner thinks,
deepseek-chat does not), so the loop's think flag is not translated.
"""
from copy import deepcopy
import json
import os
import uuid

import httpx
from ollama import ChatResponse

DEEPSEEK_URL = 'https://api.deepseek.com'
DEFAULT_MAX_TOKENS = 8192


def _turn_key(message):
    return json.dumps({'content': message.get('content'), 'tool_calls': message.get('tool_calls')}, sort_keys=True,
                      ensure_ascii=False, default=str)


class ChatCompletionsClient:
    def __init__(self, model, *, base_url=DEEPSEEK_URL, api_key=None, api_key_env='DEEPSEEK_API_KEY',
                 max_tokens=DEFAULT_MAX_TOKENS, temperature=0, timeout=600, http=None):
        key = api_key or os.environ.get(api_key_env)
        if not key and http is None:
            raise ValueError(f'No API key: set {api_key_env} or pass api_key.')
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        self.http = http or httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=10),
                                         headers={'Authorization': 'Bearer ' + key})
        self.identity = {'name': model, 'provider': base_url}
        self._turns = {}   # turn key -> list of provider tool_calls (with ids), consumed in replay order
        self.usage = []

    # --- Ollama form -> chat completions ---------------------------------------

    def _assistant(self, message, consumed):
        key = _turn_key(message)
        stored = self._turns.get(key, [])
        position = consumed.get(key, 0)
        consumed[key] = position + 1
        if position < len(stored):
            calls = deepcopy(stored[position])
        else:
            calls = [{'id': 'call_' + uuid.uuid4().hex[:24], 'type': 'function',
                      'function': {'name': c['function']['name'], 'arguments': json.dumps(c['function'].get('arguments') or {}, ensure_ascii=False)}}
                     for c in message.get('tool_calls') or []]
        out = {'role': 'assistant', 'content': message.get('content') or ''}
        if calls:
            out['tool_calls'] = calls
        return out

    def convert_messages(self, messages, format=None):
        converted, consumed, pending_ids = [], {}, []
        for message in messages:
            role = message['role']
            if role in ('system', 'user'):
                converted.append({'role': role, 'content': message['content']})
            elif role == 'assistant':
                out = self._assistant(message, consumed)
                pending_ids = [c['id'] for c in out.get('tool_calls', [])]
                converted.append(out)
            elif role == 'tool':
                if not pending_ids:
                    raise ValueError('A tool message has no preceding tool call to answer.')
                converted.append({'role': 'tool', 'tool_call_id': pending_ids.pop(0), 'content': message['content']})
            else:
                raise ValueError('Unknown message role: ' + role)
        if format:
            converted.append({'role': 'system', 'content': 'Respond with a single JSON object that matches this JSON schema and nothing else:\n'
                              + json.dumps(format, ensure_ascii=False)})
        return converted

    # --- chat completions -> Ollama form ---------------------------------------

    def _to_chat_response(self, body):
        choice = body['choices'][0]
        message = choice['message']
        calls = message.get('tool_calls') or []
        for call in calls:
            call.setdefault('id', 'call_' + uuid.uuid4().hex[:24])
        tool_calls = []
        for call in calls:
            raw = call['function'].get('arguments') or '{}'
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(arguments, dict):
                raise ValueError('Tool arguments must be a JSON object.')
            tool_calls.append({'function': {'name': call['function']['name'], 'arguments': arguments}})
        record = {'role': 'assistant', 'content': message.get('content') or '', 'thinking': message.get('reasoning_content') or None,
                  'tool_calls': tool_calls or None}
        self._turns.setdefault(_turn_key(record), []).append(calls)
        usage = body.get('usage') or {}
        self.usage.append({**usage, 'finish_reason': choice.get('finish_reason'), 'model': body.get('model')})
        return ChatResponse(model=body.get('model') or self.model, done=True,
                            done_reason='length' if choice.get('finish_reason') == 'length' else 'stop',
                            prompt_eval_count=usage.get('prompt_tokens'), eval_count=usage.get('completion_tokens'),
                            message={k: v for k, v in record.items() if v is not None})

    # --- the contract -----------------------------------------------------------------

    def chat(self, *, messages, tools=None, format=None, options=None, **ignored):
        """Ollama-shaped request in, Ollama-shaped ChatResponse out; `model`, `think`, `stream` are the client's."""
        body = {'model': self.model, 'messages': self.convert_messages(messages, format), 'max_tokens': self.max_tokens,
                'stream': False, 'temperature': (options or {}).get('temperature', self.temperature)}
        if tools:
            body['tools'] = [{'type': 'function', 'function': {k: deepcopy(t['function'][k]) for k in ('name', 'description', 'parameters')}}
                             for t in tools]
        if format:
            body['response_format'] = {'type': 'json_object'}
        response = self.http.post('/chat/completions', json=body)
        if response.status_code >= 400:
            raise ValueError(f'Chat completions error {response.status_code}: {response.text[:500]}')
        return self._to_chat_response(response.json())

    def last_usage(self):
        """The most recent request in the accounting vocabulary shared by every transport.

        DeepSeek caches prefixes on its own disk with no request-side marking
        and no separate write charge, so a hit is a cache read, a miss is
        ordinary input, and cache writes are always zero. `prompt_tokens`
        counts both, which is why the miss field is preferred when present.
        """
        record = self.usage[-1]
        hit = record.get('prompt_cache_hit_tokens') or 0
        miss = record.get('prompt_cache_miss_tokens')
        return {'input_tokens': miss if miss is not None else max(0, (record.get('prompt_tokens') or 0) - hit),
                'output_tokens': record.get('completion_tokens') or 0,
                'cache_read_tokens': hit, 'cache_write_tokens': 0}

    def close(self):
        self.http.close()

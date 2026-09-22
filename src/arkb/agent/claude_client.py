"""Claude Messages API behind the agent loop's Ollama-shaped chat(**request) contract.

The loop speaks one dialect: `messages` in Ollama form (assistant messages
carry `content`, `thinking` and `tool_calls`; tool results are `role: tool`
messages), `tools` as function definitions, `format` as a JSON schema for the
reserved finalization, and a `think` flag. This client translates that dialect
to the Messages API and back, so the loop, the session and the observer stay
provider-neutral.

Two translations need care. Assistant turns must be replayed with the blocks
Claude produced (thinking blocks and tool_use ids), which the Ollama form does
not carry, so the client keeps them by turn and restores them in order. And the
loop's `think=False` (its finalization) is not "thinking off": on Claude it is
adaptive thinking at low effort, which avoids the failure modes of disabled
thinking and still keeps the request cheap.

Prompt caching. Every turn resends the whole conversation, so uncached input is
the dominant cost. Caching is a prefix match over the rendered request, whose
order is tools, then system, then messages, with at most four `cache_control`
breakpoints; a byte change anywhere before a breakpoint invalidates it. This
client marks three stable prefixes and nothing else: the last tool definition,
the system block, and — in a multi-turn session — the end of the conversation
carried over from earlier turns. Everything the current turn produces comes
after the last breakpoint, so no marker ever sits on content that changes
within a run. The deliberate omission is a breakpoint that moves with each
tool observation: it would cache more inside one run, at the price of a prefix
that is no longer stable across turns, which is the property tests can check
offline. One invalidation is inherent to the loop and is not worked around
here: the reserved finalization both drops `tools` and lowers the effort, so
that request starts from a cold prefix.
"""
from copy import deepcopy
import json
import uuid

from ollama import ChatResponse

DEFAULT_MODEL = 'claude-opus-5'
DEFAULT_MAX_TOKENS = 16000
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
CACHE_CONTROL = {'type': 'ephemeral'}
# Content-block types that accept cache_control; a thinking block does not.
CACHEABLE_BLOCKS = ('text', 'tool_use', 'tool_result', 'image', 'document')
MAX_BREAKPOINTS = 4
# Keywords the structured-output grammar does not accept; the session validates the full schema afterwards.
_UNSUPPORTED_FORMAT_KEYWORDS = {'minLength', 'maxLength', 'uniqueItems', 'minItems', 'maxItems', 'minimum', 'maximum',
                                'pattern', 'default'}


def _turn_key(message):
    return json.dumps({'content': message.get('content'), 'tool_calls': message.get('tool_calls')}, sort_keys=True,
                      ensure_ascii=False, default=str)


def output_schema(schema):
    """The finalization schema without keywords the structured-output grammar rejects."""
    if isinstance(schema, dict):
        return {k: output_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED_FORMAT_KEYWORDS}
    if isinstance(schema, list):
        return [output_schema(v) for v in schema]
    return schema


def convert_tools(tools, *, cache=False):
    """Translate the loop's function definitions; cache marks the last one as a read point.

    The order is the session's, which is fixed for a run, and each schema is
    copied verbatim, so the rendered tool block is byte-identical every turn.
    """
    converted = [{'name': t['function']['name'], 'description': t['function']['description'],
                  'input_schema': deepcopy(t['function']['parameters'])} for t in tools or []]
    if cache and converted:
        converted[-1]['cache_control'] = dict(CACHE_CONTROL)
    return converted


def mark_cacheable(blocks):
    """Put a breakpoint on the last block that accepts one; False when none does."""
    for block in reversed(blocks):
        if block.get('type') in CACHEABLE_BLOCKS:
            block['cache_control'] = dict(CACHE_CONTROL)
            return True
    return False


def cache_breakpoints(request):
    """Count the breakpoints in a rendered request, in the order the API renders them."""
    blocks = list(request.get('tools') or []) + list(request.get('system') or [])
    for message in request.get('messages') or []:
        content = message.get('content')
        blocks += content if isinstance(content, list) else []
    return sum('cache_control' in block for block in blocks)


class ClaudeClient:
    def __init__(self, model=DEFAULT_MODEL, *, effort='high', low_effort='low', max_tokens=DEFAULT_MAX_TOKENS,
                 client=None, api_key=None, cache=True):
        if effort not in EFFORTS or low_effort not in EFFORTS:
            raise ValueError('effort must be one of ' + ', '.join(EFFORTS))
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model, self.effort, self.low_effort, self.max_tokens, self.client = model, effort, low_effort, max_tokens, client
        self.cache = cache
        self.identity = {'name': model, 'provider': 'anthropic'}
        self._turns = {}   # turn key -> list of original assistant content blocks, consumed in replay order
        self.usage = []    # per-request usage records for accounting

    # --- Ollama form -> Messages API ----------------------------------------

    def _assistant_blocks(self, message, consumed):
        key = _turn_key(message)
        stored = self._turns.get(key, [])
        position = consumed.get(key, 0)
        consumed[key] = position + 1
        if position < len(stored):
            return deepcopy(stored[position])
        # A turn this client did not produce (or a resumed conversation): rebuild without thinking.
        blocks = []
        if message.get('content'):
            blocks.append({'type': 'text', 'text': message['content']})
        for call in message.get('tool_calls') or []:
            function = call['function']
            blocks.append({'type': 'tool_use', 'id': 'toolu_' + uuid.uuid4().hex[:24], 'name': function['name'],
                           'input': function.get('arguments') or {}})
        return blocks or [{'type': 'text', 'text': '(no content)'}]

    def convert_messages(self, messages):
        """Top-level system text, Messages-API turns, and where the carried history ends.

        Tool results follow their tool_use ids in order. The boundary is the
        index of the last converted message that precedes this turn's query:
        everything up to it was fixed before the turn began, which makes it the
        one position in `messages` worth a cache breakpoint. It is None for a
        single-turn run, where nothing precedes the query.
        """
        system, converted, consumed, pending_ids = [], [], {}, []
        users = [i for i, m in enumerate(messages) if m['role'] == 'user']
        carried, boundary = users[-1] if users else len(messages), None
        for index, message in enumerate(messages):
            if index == carried and converted:
                boundary = len(converted) - 1
            role = message['role']
            if role == 'system':
                if not converted:
                    system.append(message['content'])
                else:
                    # An operator instruction mid-conversation travels as text in the next user turn.
                    converted.append({'role': 'user', 'content': [{'type': 'text', 'text': message['content']}]})
            elif role == 'user':
                converted.append({'role': 'user', 'content': [{'type': 'text', 'text': message['content']}]})
            elif role == 'assistant':
                blocks = self._assistant_blocks(message, consumed)
                pending_ids = [b['id'] for b in blocks if b.get('type') == 'tool_use']
                converted.append({'role': 'assistant', 'content': blocks})
            elif role == 'tool':
                if not pending_ids:
                    raise ValueError('A tool message has no preceding tool call to answer.')
                block = {'type': 'tool_result', 'tool_use_id': pending_ids.pop(0), 'content': message['content']}
                if converted and converted[-1]['role'] == 'user' and converted[-1]['content'] and \
                        converted[-1]['content'][-1].get('type') == 'tool_result':
                    converted[-1]['content'].append(block)
                else:
                    converted.append({'role': 'user', 'content': [block]})
            else:
                raise ValueError('Unknown message role: ' + role)
        return '\n\n'.join(system), converted, boundary

    # --- Messages API -> Ollama form ----------------------------------------

    def _to_chat_response(self, response):
        blocks = [b.model_dump(exclude_none=True) if hasattr(b, 'model_dump') else dict(b) for b in response.content]
        text = ''.join(b['text'] for b in blocks if b.get('type') == 'text')
        thinking = ''.join(b.get('thinking') or '' for b in blocks if b.get('type') == 'thinking') or None
        tool_calls = [{'function': {'name': b['name'], 'arguments': b['input']}} for b in blocks if b.get('type') == 'tool_use']
        message = {'role': 'assistant', 'content': text, 'thinking': thinking, 'tool_calls': tool_calls or None}
        self._turns.setdefault(_turn_key(message), []).append(blocks)
        stop = response.stop_reason
        done_reason = 'length' if stop == 'max_tokens' else 'stop'
        usage = response.usage
        record = {'input_tokens': usage.input_tokens, 'output_tokens': usage.output_tokens,
                  'cache_read_input_tokens': getattr(usage, 'cache_read_input_tokens', None),
                  'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', None),
                  'stop_reason': stop, 'model': response.model}
        self.usage.append(record)
        return ChatResponse(model=response.model, done=True, done_reason=done_reason,
                            prompt_eval_count=usage.input_tokens, eval_count=usage.output_tokens,
                            message={k: v for k, v in message.items() if v is not None})

    # --- the contract -------------------------------------------------------

    def chat(self, *, messages, think=False, tools=None, format=None, **ignored):
        """Ollama-shaped request in, Ollama-shaped ChatResponse out.

        `model`, `options`, `stream`, `truncate` and `shift` in the request
        describe the local transport and are ignored here: the model is this
        client's, sampling parameters do not exist on current Claude models,
        and the context is the model's own.
        """
        system, converted, boundary = self.convert_messages(messages)
        kwargs = {'model': self.model, 'max_tokens': self.max_tokens, 'messages': converted,
                  'thinking': {'type': 'adaptive'},
                  'output_config': {'effort': self.effort if think else self.low_effort}}
        if system:
            kwargs['system'] = [{'type': 'text', 'text': system}]
            if self.cache:
                kwargs['system'][0]['cache_control'] = dict(CACHE_CONTROL)
        if tools:
            kwargs['tools'] = convert_tools(tools, cache=self.cache)
        if self.cache and boundary is not None:
            mark_cacheable(converted[boundary]['content'])
        if format:
            kwargs['output_config']['format'] = {'type': 'json_schema', 'schema': output_schema(format)}
        if cache_breakpoints(kwargs) > MAX_BREAKPOINTS:
            raise ValueError('A request may carry at most four cache_control breakpoints.')
        response = self.client.messages.create(**kwargs)
        if response.stop_reason == 'refusal':
            raise ValueError('The model declined the request: ' + str(getattr(response, 'stop_details', None)))
        return self._to_chat_response(response)

    def last_usage(self):
        """The most recent request in the accounting vocabulary shared by every transport.

        input_tokens is already the uncached remainder on this API; the whole
        prompt is that plus the cache read and the cache write.
        """
        record = self.usage[-1]
        return {'input_tokens': record['input_tokens'] or 0, 'output_tokens': record['output_tokens'] or 0,
                'cache_read_tokens': record['cache_read_input_tokens'] or 0,
                'cache_write_tokens': record['cache_creation_input_tokens'] or 0}

    def close(self):
        close = getattr(self.client, 'close', None)
        if close:
            close()

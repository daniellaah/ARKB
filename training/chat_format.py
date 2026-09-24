"""One rendering of a conversation, shared by the transport and the dataset builder.

The student is served by `mlx_lm.server`, which turns an OpenAI-shaped request
into tokens with `tokenizer.apply_chat_template(messages, tools=...)`. A
training sample must therefore be the same template applied to the same
messages, or the model is trained on a prompt it will never see. That is the
silent failure this module exists to prevent: `to_wire` is the only place a
conversation becomes a request body, and both the transport that talks to the
server and the builder that renders training samples call it.

Three adaptations happen here, each needed for Qwen3.5's template and each
applied identically on both sides:

- A system message that is not the first message raises in the template
  ("System message must be at the beginning"). The agent loop appends system
  messages mid-conversation (the stall reminder, the finalization
  instruction), so those become user messages.
- Tool arguments travel as the JSON strings the OpenAI wire format uses, and
  `mlx_lm.server` parses them back into dicts before templating, because the
  template iterates over the argument object. `for_template` is that same
  parse, so the builder renders what the server renders rather than something
  equivalent-looking.
- A parameter typed `["string", "null"]` in our schemas is advertised as
  `"string"`. The XML tool-call format Qwen3.5 emits carries no types, so the
  server's parser recovers them from the advertised schema; a union type it
  does not recognize falls through to `ast.literal_eval`, which turns a source
  path into a parse error and drops the whole call.
"""

from copy import deepcopy
import json

MODEL_PATH = 'mlx-community/Qwen3.5-4B-bf16'

SCHEMA_INSTRUCTION = ('Respond with a single JSON object that matches this JSON schema and nothing else:\n')


def wire_tools(tools):
    """Advertise one concrete type per parameter; see the module note on union types."""
    if not tools:
        return None
    out = deepcopy(list(tools))
    for tool in out:
        for spec in (tool.get('function', {}).get('parameters', {}).get('properties') or {}).values():
            kind = spec.get('type')
            if isinstance(kind, list):
                concrete = [k for k in kind if k != 'null']
                spec['type'] = concrete[0] if concrete else 'string'
            if 'enum' in spec:
                spec['enum'] = [v for v in spec['enum'] if v is not None]
    return out


def to_wire(messages, format=None):
    """The Ollama-shaped conversation as the server receives it, ready for the template."""
    out, pending = [], []
    for message in messages:
        role = message['role']
        if role in ('system', 'user'):
            # Only the first message may be a system message; see the module note.
            out.append({'role': 'system' if role == 'system' and not out else 'user',
                        'content': message['content']})
        elif role == 'assistant':
            calls = [{'id': f'call_{len(out)}_{index}', 'type': 'function',
                      'function': {'name': call['function']['name'],
                                   'arguments': json.dumps(call['function'].get('arguments') or {},
                                                           ensure_ascii=False)}}
                     for index, call in enumerate(message.get('tool_calls') or [])]
            entry = {'role': 'assistant', 'content': message.get('content') or ''}
            if message.get('thinking'):
                entry['reasoning_content'] = message['thinking']
            if calls:
                entry['tool_calls'] = calls
            pending = [call['id'] for call in calls]
            out.append(entry)
        elif role == 'tool':
            if not pending:
                raise ValueError('A tool message has no preceding tool call to answer.')
            out.append({'role': 'tool', 'tool_call_id': pending.pop(0), 'content': message['content']})
        else:
            raise ValueError('Unknown message role: ' + role)
    if format:
        out.append({'role': 'user', 'content': SCHEMA_INSTRUCTION + json.dumps(format, ensure_ascii=False)})
    return out


def for_template(messages):
    """What `mlx_lm.server.process_message_content` leaves for the chat template."""
    out = deepcopy(messages)
    for message in out:
        for call in message.get('tool_calls') or []:
            arguments = call.get('function', {}).get('arguments')
            if isinstance(arguments, str) and arguments:
                call['function']['arguments'] = json.loads(arguments)
    return out


def render(tokenizer, messages, tools, *, add_generation_prompt, enable_thinking=True):
    """Token ids for one conversation, exactly as `mlx_lm.server` would build them."""
    return tokenizer.apply_chat_template(
        for_template(messages), tools=tools, add_generation_prompt=add_generation_prompt,
        tokenize=True, enable_thinking=enable_thinking)


def render_text(tokenizer, messages, tools, *, add_generation_prompt, enable_thinking=True):
    """The same rendering before tokenization, for comparing two renderings as text."""
    return tokenizer.apply_chat_template(
        for_template(messages), tools=tools, add_generation_prompt=add_generation_prompt,
        tokenize=False, enable_thinking=enable_thinking)

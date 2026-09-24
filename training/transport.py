"""Two transports the pilot needs and the product does not: a local student, a sampled teacher.

Both are the product's `ChatCompletionsClient` with the smallest possible
change, so the agent loop, the tool session and the observer are the ones the
evaluation runs. Nothing here is imported by `src/arkb`.

`MlxServerClient` points that client at `mlx_lm.server`, an OpenAI-compatible
endpoint serving the LoRA-tuned student. It differs from the DeepSeek path in
four places, all of them properties of Qwen3.5's chat template rather than of
the endpoint: the conversation is rendered by `chat_format.to_wire`, the tool
schemas by `chat_format.wire_tools`, thinking is switched off for the reserved
finalization the way Ollama's `think=False` switches it off, and the server
reports reasoning under `reasoning` where DeepSeek uses `reasoning_content`.

`SampledDeepSeekClient` exists because the loop pins `temperature: 0` in every
request, which is right for an evaluation and wrong for generating a training
set: four identical samples of one question teach nothing. It overrides that
one option and leaves the rest of the request alone.
"""

import json

from arkb.agent.chat_completions_client import ChatCompletionsClient

from training import chat_format

MLX_SERVER_URL = 'http://127.0.0.1:8080/v1'


class MlxServerClient(ChatCompletionsClient):
    """The student, served locally; `label` is what the run records as the model."""

    def __init__(self, label, *, base_url=MLX_SERVER_URL, max_tokens=4096, temperature=0, timeout=900):
        super().__init__(label, base_url=base_url, api_key='local', max_tokens=max_tokens,
                         temperature=temperature, timeout=timeout)
        self.identity = {'name': label, 'provider': base_url}

    def convert_messages(self, messages, format=None):
        return chat_format.to_wire(messages, format)

    def chat(self, *, messages, tools=None, format=None, options=None, **ignored):
        body = {'model': 'default_model', 'messages': self.convert_messages(messages, format),
                'max_tokens': self.max_tokens, 'stream': False,
                'temperature': (options or {}).get('temperature', self.temperature)}
        if tools:
            body['tools'] = chat_format.wire_tools(tools)
        if format:
            # The reserved finalization formats an answer from evidence already
            # collected; the loop turns thinking off for it, and this is how the
            # template is told. Structured output is not enforced here, so the
            # schema travels as the instruction message `to_wire` appends.
            body['chat_template_kwargs'] = {'enable_thinking': False}
        response = self.http.post('/chat/completions', json=body)
        if response.status_code >= 400:
            raise ValueError(f'Chat completions error {response.status_code}: {response.text[:500]}')
        payload = response.json()
        message = payload['choices'][0]['message']
        if 'reasoning' in message:
            message.setdefault('reasoning_content', message.pop('reasoning'))
        return self._to_chat_response(payload)


class SampledDeepSeekClient(ChatCompletionsClient):
    """The teacher, sampled: this client's temperature wins over the request's."""

    def __init__(self, model, *, temperature, **kwargs):
        super().__init__(model, temperature=temperature, **kwargs)

    def chat(self, *, options=None, **request):
        return super().chat(options={**(options or {}), 'temperature': self.temperature}, **request)


def tool_definitions_sent(request):
    """The tool schemas a saved request carried, in the form the template renders."""
    return chat_format.wire_tools(request.get('tools')) if request.get('tools') else None


def dump(value):
    return json.dumps(value, ensure_ascii=False)

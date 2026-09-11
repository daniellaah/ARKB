"""Scripted Ollama responses; no query routing or real model inference."""

from copy import deepcopy

from ollama import ChatResponse


def tool_call(name, **arguments):
    return {'function': {'name': name, 'arguments': arguments}}


def reply(content=None, *, calls=(), done_reason='stop'):
    return ChatResponse(message={'role': 'assistant', 'content': content, 'tool_calls': list(calls)},
                        done=True, done_reason=done_reason)


class ScriptedModel:
    def __init__(self, *steps):
        self.steps = iter(steps)
        self.requests = []

    def chat(self, **request):
        self.requests.append(deepcopy(request))
        step = next(self.steps)
        return step(request['messages']) if callable(step) else step


def complete(answer='answer', *, status=None, constrained=False):
    """A scripted final proposal citing all delivered evidence, not model policy."""
    import json
    def respond(messages):
        refs = []
        for message in messages:
            if message['role'] != 'tool':
                continue
            result = json.loads(message['content'])
            hits = [result['result']] if 'result' in result else result.get('results', [])
            for hit in hits:
                if hit['ref'] not in refs:
                    refs.append(hit['ref'])
        args = {'answer': answer, 'status': status or ('answered' if refs else 'insufficient_evidence'),
                'evidence_refs': refs}
        return reply(json.dumps(args)) if constrained else reply(calls=[tool_call('finish', **args)])
    return respond

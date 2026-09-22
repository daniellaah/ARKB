"""Direct Ollama transport for evaluation runs: explicit model, options and thinking; nothing hidden.

Ollama accepts truncate=false and shift=false on /api/chat; both are always sent
so a request never silently loses context. The caller's think flag wins (the
product loop finalizes without thinking on purpose), and a request for another
model is refused rather than rewritten, so the recorded trace is what was sent.
"""
import hashlib
import json

import httpx
from ollama import ChatResponse

OLLAMA_URL = 'http://127.0.0.1:11434'


def model_identity(http, name):
    """Digest, details and a definition hash of the installed model, for run.json."""
    tags = http.get('/api/tags').raise_for_status().json()['models']
    matches = [m for m in tags if m['name'] == name]
    if len(matches) != 1:
        raise ValueError('Required model unavailable: ' + name)
    show = http.post('/api/show', json={'model': name}).raise_for_status().json()
    definition = {k: show.get(k) for k in ('template', 'parameters', 'capabilities', 'model_info', 'details')}
    return {'name': name, 'digest': matches[0]['digest'], 'details': matches[0]['details'],
            'definition_sha256': hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest(),
            'template_sha256': hashlib.sha256((show.get('template') or '').encode()).hexdigest(),
            'parameters': show.get('parameters')}


class OllamaClient:
    def __init__(self, model, *, options, think, url=OLLAMA_URL, timeout=330):
        self.model, self.options, self.think = model, dict(options), think
        self.http = httpx.Client(base_url=url, timeout=httpx.Timeout(timeout, connect=10))
        self.identity = model_identity(self.http, model)

    def chat(self, **request):
        if request.get('model', self.model) != self.model:
            raise ValueError('Request model differs from the client model.')
        request = dict(request, model=self.model, think=request.get('think', self.think), truncate=False, shift=False,
                       options={**(request.get('options') or {}), **self.options})
        response = self.http.post('/api/chat', json=request)
        response.raise_for_status()
        value = response.json()
        if 'error' in value:
            raise ValueError('Ollama response error: ' + value['error'])
        return ChatResponse.model_validate(value)

    def close(self):
        self.http.close()

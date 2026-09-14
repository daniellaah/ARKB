"""Direct local Ollama transport preserves fields absent from the installed SDK.

Ollama 0.33.2 supports truncate=false and shift=false on /api/chat. Both are
explicitly recorded before dispatch. HTTP and response parsing failures remain
visible to the unchanged product Agent loop.
"""
from copy import deepcopy
import hashlib
import json

import httpx
from ollama import ChatResponse

from .contract import MODEL, OPTIONS, THINK


def model_identity(http, name):
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


class LocalChatClient:
    def __init__(self, expected_identity, *, journal=None):
        self.http = httpx.Client(base_url='http://127.0.0.1:11434', timeout=httpx.Timeout(330, connect=10))
        self.identity, self.journal = expected_identity, journal

    def verify_identity(self):
        if model_identity(self.http, MODEL) != self.identity:
            raise ValueError('Chat model identity changed.')

    def chat(self, **request):
        if (request['model'] != MODEL or request['options'] != OPTIONS or request['think'] is not THINK
                or request.get('truncate') is not False or request.get('shift') is not False):
            raise ValueError('Request escaped the registered model/options/context contract.')
        if self.journal:
            self.journal({'event': 'provider_request', 'request': deepcopy(request)})
        response = self.http.post('/api/chat', json=request)
        if self.journal:
            self.journal({'event': 'provider_response', 'http_status': response.status_code, 'body': response.text})
        response.raise_for_status()
        value = response.json()
        if 'error' in value:
            raise ValueError('Ollama response error: ' + value['error'])
        return ChatResponse.model_validate(value)

    def close(self):
        self.http.close()

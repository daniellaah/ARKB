"""Choose a chat transport for a model name and account for what a run costs.

The agent loop speaks one dialect: Ollama's `chat(**request)`. Three transports
implement it (a local Ollama server, the Claude Messages API, an
OpenAI-compatible endpoint). The selection rule lives here so the product CLI
and the evaluation harness cannot drift apart: `claude-*` goes to the Messages
API, `deepseek-*` to the chat-completions endpoint, everything else to Ollama.

Accounting lives here too, because only a transport knows how its provider
reports cached input. ChatUsage wraps any of them and normalizes one row per
request into four fields: input tokens billed at the full rate, cache reads,
cache writes and output. Prices are list prices recorded as constants with the
date they were read; an estimate is an estimate, not an invoice.
"""

import os

import httpx
from ollama import ChatResponse

OLLAMA_URL = 'http://127.0.0.1:11434'
HOSTED_PREFIXES = ('claude', 'deepseek')

# List prices in USD per million tokens as (input, cache write, cache read, output).
# Anthropic, read 2026-09-22 from https://platform.claude.com/docs/en/about-claude/pricing:
# the cache-write column is the 5-minute write (1.25x input), which is the TTL this
# client uses. DeepSeek, read 2026-09-22 from https://api-docs.deepseek.com/quick_start/pricing:
# the input column is its cache-miss rate and the read column its cache-hit rate; DeepSeek
# caches server-side and charges nothing extra to write, so the write column repeats the
# miss rate and is never reached. DeepSeek quotes peak and off-peak rates, off-peak being
# exactly half; these are the peak rates, so a DeepSeek estimate is an upper bound.
# `deepseek-chat` and `deepseek-reasoner` are the legacy names for the non-thinking and
# thinking modes of DeepSeek's flash model and are priced with it.
PRICES_CHECKED = '2026-09-22'
PRICES = {
    'claude-fable-5-1': (10.0, 12.50, 0.25, 50.0),
    'claude-fable-5': (10.0, 12.50, 1.00, 50.0),
    'claude-opus-5-5': (4.0, 5.00, 0.20, 20.0),
    'claude-opus-5': (5.0, 6.25, 0.50, 25.0),
    'claude-opus-4-8': (5.0, 6.25, 0.50, 25.0),
    'claude-opus-4-7': (5.0, 6.25, 0.50, 25.0),
    'claude-opus-4-6': (5.0, 6.25, 0.50, 25.0),
    'claude-opus-4-5': (5.0, 6.25, 0.50, 25.0),
    'claude-sonnet-5': (2.0, 2.50, 0.20, 10.0),
    'claude-sonnet-4-6': (3.0, 3.75, 0.30, 15.0),
    'claude-sonnet-4-5': (3.0, 3.75, 0.30, 15.0),
    'claude-haiku-4-5': (1.0, 1.25, 0.10, 5.0),
    'deepseek-flash': (0.30, 0.30, 0.006, 1.20),
    'deepseek-chat': (0.30, 0.30, 0.006, 1.20),
    'deepseek-reasoner': (0.30, 0.30, 0.006, 1.20),
    'deepseek-v4-pro': (1.32, 1.32, 0.044, 3.96),
}
USAGE_FIELDS = ('input_tokens', 'cache_read_tokens', 'cache_write_tokens', 'output_tokens')


def _count(value) -> int:
    """A provider counter, or zero when it reports none; accounting never fails a run."""
    return value if type(value) is int and value >= 0 else 0


class OllamaClient:
    """Sends exactly what the loop asks (its think flag wins) plus the context and output limits."""

    def __init__(self, model, *, options=None, think=False, host=OLLAMA_URL):
        self.model, self.options, self.think = model, dict(options or {}), think
        self.http = httpx.Client(base_url=host, timeout=httpx.Timeout(330, connect=10))
        tags = self.http.get('/api/tags').raise_for_status().json()['models']
        match = [m for m in tags if m['name'] == model]
        if len(match) != 1:
            raise ValueError('Model unavailable in Ollama: ' + model)
        self.identity = {'name': model, 'digest': match[0]['digest']}

    def chat(self, **request):
        if request.get('model', self.model) != self.model:
            raise ValueError('Request model differs from the client model.')
        request = dict(request, model=self.model, think=request.get('think', self.think), truncate=False, shift=False,
                       options={**(request.get('options') or {}), **self.options})
        value = self.http.post('/api/chat', json=request).raise_for_status().json()
        if 'error' in value:
            raise ValueError('Ollama response error: ' + value['error'])
        return ChatResponse.model_validate(value)

    def close(self):
        self.http.close()


def is_hosted(model: str) -> bool:
    """A model served by a paid API rather than by the local Ollama server."""
    return model.startswith(HOSTED_PREFIXES)


def hosted_client(model: str, *, options=None, think: bool = False, effort: str = 'high'):
    """Build the hosted transport for a hosted model name; None for a local one.

    A missing key is reported here, before any request, because the provider
    SDKs report it as their own error type at an unhelpful moment.
    """
    options = options or {}
    if model.startswith('claude'):
        if not (os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('ANTHROPIC_AUTH_TOKEN')):
            raise ValueError(f'{model} needs ANTHROPIC_API_KEY; set it in the environment or in .env at the '
                             'repository root.')
        from arkb.agent.claude_client import ClaudeClient
        try:
            # The SDK is imported when the client is built, so a missing extra surfaces here.
            return ClaudeClient(model, effort=effort if think else 'low')
        except ImportError as error:
            raise ValueError('Claude models need their optional dependency: '
                             'uv sync --extra claude (or pip install "arkb[claude]").') from error
    if model.startswith('deepseek'):
        if not os.environ.get('DEEPSEEK_API_KEY'):
            raise ValueError(f'{model} needs DEEPSEEK_API_KEY; set it in the environment or in .env at the '
                             'repository root.')
        from arkb.agent.chat_completions_client import ChatCompletionsClient
        return ChatCompletionsClient(model, temperature=options.get('temperature', 0))
    return None


def make_client(model, *, options, think, effort='high'):
    """Ollama for local models; Claude for claude-* (think selects the effort); DeepSeek for deepseek-* (DEEPSEEK_API_KEY)."""
    return hosted_client(model, options=options, think=think, effort=effort) or \
        OllamaClient(model, options=options, think=think)


class ChatUsage:
    """Wrap any transport and record one normalized usage row per request.

    This adds nothing to the loop's contract: it forwards `chat(**request)` and
    returns the transport's own response. A hosted transport keeps a
    provider-shaped usage list and reports the row through `last_usage()`;
    otherwise the Ollama-shaped response counters are the whole story and the
    prompt is uncached by definition. Lifetime stays with whoever opened the
    wrapped client, so this wrapper deliberately has no close().
    """

    def __init__(self, client, model: str | None = None):
        self.client = client
        self.model = model or (getattr(client, 'identity', None) or {}).get('name')
        self.records: list[dict[str, int]] = []

    def chat(self, **request):
        response = self.client.chat(**request)
        report = getattr(self.client, 'last_usage', None)
        row = report() if report is not None else {
            'input_tokens': response.prompt_eval_count, 'output_tokens': response.eval_count,
            'cache_read_tokens': 0, 'cache_write_tokens': 0}
        self.records.append({key: _count(row.get(key)) for key in USAGE_FIELDS})
        return response

    def totals(self, start: int = 0) -> dict:
        """Sum the rows recorded since `start`; prompt_tokens is the whole prompt, cached or not."""
        rows = self.records[start:]
        totals = {key: sum(row[key] for row in rows) for key in USAGE_FIELDS}
        return {'model': self.model, 'requests': len(rows), **totals,
                'prompt_tokens': totals['input_tokens'] + totals['cache_read_tokens'] + totals['cache_write_tokens']}


def estimate_cost(totals: dict) -> float | None:
    """List-price estimate in USD, or None when no price is recorded for the model."""
    price = PRICES.get(totals.get('model'))
    if price is None:
        return None
    base, write, read, output = price
    return (totals['input_tokens'] * base + totals['cache_write_tokens'] * write
            + totals['cache_read_tokens'] * read + totals['output_tokens'] * output) / 1e6


def format_usage(totals: dict, *, label: str = 'usage') -> list[str]:
    """One accounting line, plus one cost line for a hosted model."""
    lines = [f"{label}: {totals['requests']} request(s) | prompt {totals['prompt_tokens']:,} "
             f"(uncached {totals['input_tokens']:,}, cache read {totals['cache_read_tokens']:,}, "
             f"cache write {totals['cache_write_tokens']:,}) | output {totals['output_tokens']:,}"]
    model = totals.get('model') or ''
    if not is_hosted(model):
        return lines
    cost = estimate_cost(totals)
    if cost is None:
        lines.append(f'cost: unknown, no list price recorded for {model} (table checked {PRICES_CHECKED})')
    else:
        upper = ', peak rate: off-peak is half' if model.startswith('deepseek') else ''
        lines.append(f'cost: ~${cost:.4f} at {model} list prices checked {PRICES_CHECKED}{upper}')
    return lines

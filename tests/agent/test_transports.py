"""Transport selection and run accounting; no provider is contacted."""

from types import SimpleNamespace

import pytest

from arkb.agent import transports
from arkb.agent.transports import ChatUsage, estimate_cost, format_usage, is_hosted, make_client


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for name in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'DEEPSEEK_API_KEY'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def built(monkeypatch):
    """Record what each transport class would be constructed with."""
    calls = []

    def record(kind):
        def build(model, **kwargs):
            calls.append((kind, model, kwargs))
            return SimpleNamespace(identity={'name': model}, kind=kind)
        return build

    monkeypatch.setattr('arkb.agent.claude_client.ClaudeClient', record('claude'))
    monkeypatch.setattr('arkb.agent.chat_completions_client.ChatCompletionsClient', record('deepseek'))
    monkeypatch.setattr(transports, 'OllamaClient', record('ollama'))
    return calls


def test_the_model_name_selects_the_transport_and_think_selects_the_effort(built, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
    options = {'temperature': 0, 'num_ctx': 32768}
    assert make_client('qwen3.5:9b', options=options, think=True).kind == 'ollama'
    assert make_client('claude-opus-5', options=options, think=True, effort='max').kind == 'claude'
    assert make_client('claude-opus-5', options=options, think=False, effort='max').kind == 'claude'
    assert make_client('deepseek-reasoner', options=options, think=True).kind == 'deepseek'
    assert [(kind, model) for kind, model, _ in built] == [
        ('ollama', 'qwen3.5:9b'), ('claude', 'claude-opus-5'), ('claude', 'claude-opus-5'),
        ('deepseek', 'deepseek-reasoner')]
    assert built[0][2] == {'options': options, 'think': True}
    # think chooses between the caller's effort and the cheap one; the loop's own think flag still wins per request.
    assert built[1][2] == {'effort': 'max'} and built[2][2] == {'effort': 'low'}
    assert built[3][2] == {'temperature': 0}
    assert is_hosted('claude-opus-5') and is_hosted('deepseek-chat') and not is_hosted('qwen3.5:9b')


@pytest.mark.parametrize('model,variable', [('claude-opus-5', 'ANTHROPIC_API_KEY'),
                                            ('deepseek-reasoner', 'DEEPSEEK_API_KEY')])
def test_a_missing_key_is_reported_before_any_request(built, model, variable):
    with pytest.raises(ValueError, match=variable):
        make_client(model, options={}, think=True)
    assert built == []


def test_an_anthropic_oauth_token_also_counts_as_a_key(built, monkeypatch):
    monkeypatch.setenv('ANTHROPIC_AUTH_TOKEN', 'test')
    assert make_client('claude-opus-5', options={}, think=True).kind == 'claude'


class Scripted:
    """A transport whose provider reports cached input; the loop's contract, nothing more."""

    def __init__(self, *rows, identity=None):
        self.rows, self.identity, self.requests = list(rows), identity or {'name': 'claude-opus-5'}, []

    def chat(self, **request):
        self.requests.append(request)
        self.last = self.rows.pop(0)
        return SimpleNamespace(prompt_eval_count=None, eval_count=None)

    def last_usage(self):
        return self.last


def test_usage_is_summed_per_request_and_can_be_read_back_for_one_turn():
    row = {'input_tokens': 100, 'output_tokens': 20, 'cache_read_tokens': 900, 'cache_write_tokens': 80}
    client = ChatUsage(Scripted(row, row, row))
    client.chat(messages=[])
    mark = len(client.records)
    client.chat(messages=[])
    client.chat(messages=[])
    assert client.model == 'claude-opus-5'
    session, turn = client.totals(), client.totals(mark)
    assert (session['requests'], turn['requests']) == (3, 2)
    assert session['input_tokens'] == 300 and session['cache_read_tokens'] == 2700
    # The reported prompt is the whole prompt: uncached, read from cache and written to it.
    assert session['prompt_tokens'] == 3 * 1080 and turn['prompt_tokens'] == 2 * 1080
    assert client.client.requests == [{'messages': []}] * 3


def test_a_transport_without_cache_accounting_falls_back_to_the_response_counters():
    response = SimpleNamespace(prompt_eval_count=1200, eval_count=340)
    client = ChatUsage(SimpleNamespace(chat=lambda **request: response), model='qwen3.5:9b')
    client.chat(messages=[])
    totals = client.totals()
    assert totals['input_tokens'] == 1200 and totals['prompt_tokens'] == 1200
    assert totals['cache_read_tokens'] == 0 and totals['cache_write_tokens'] == 0
    assert format_usage(totals) == [
        'usage: 1 request(s) | prompt 1,200 (uncached 1,200, cache read 0, cache write 0) | output 340']


def test_counters_a_provider_omits_do_not_break_accounting():
    client = ChatUsage(SimpleNamespace(chat=lambda **request: SimpleNamespace(
        prompt_eval_count=None, eval_count=-1)), model='qwen3.5:9b')
    client.chat(messages=[])
    assert client.totals()['prompt_tokens'] == 0 and client.totals()['output_tokens'] == 0


def test_cost_uses_the_recorded_list_prices_and_charges_cached_input_less():
    million = {'input_tokens': 1_000_000, 'cache_read_tokens': 0, 'cache_write_tokens': 0,
               'output_tokens': 0, 'requests': 1, 'prompt_tokens': 1_000_000}
    assert estimate_cost({**million, 'model': 'claude-opus-5'}) == 5.0
    assert estimate_cost({**million, 'model': 'claude-opus-5', 'input_tokens': 0,
                          'cache_read_tokens': 1_000_000}) == 0.5
    assert estimate_cost({**million, 'model': 'claude-opus-5', 'input_tokens': 0,
                          'cache_write_tokens': 1_000_000}) == 6.25
    assert estimate_cost({**million, 'model': 'claude-opus-5', 'input_tokens': 0,
                          'output_tokens': 1_000_000}) == 25.0
    assert estimate_cost({**million, 'model': 'deepseek-reasoner'}) == 0.30
    assert estimate_cost({**million, 'model': 'qwen3.5:9b'}) is None


def test_only_a_hosted_model_reports_a_cost_and_an_unpriced_one_says_so():
    totals = {'model': 'qwen3.5:9b', 'requests': 1, 'prompt_tokens': 10, 'input_tokens': 10,
              'cache_read_tokens': 0, 'cache_write_tokens': 0, 'output_tokens': 5}
    assert len(format_usage(totals)) == 1
    priced = format_usage({**totals, 'model': 'claude-opus-5'})
    assert len(priced) == 2 and priced[1].startswith('cost: ~$0.0002 at claude-opus-5 list prices checked ')
    assert transports.PRICES_CHECKED in priced[1]
    upper = format_usage({**totals, 'model': 'deepseek-reasoner'})[1]
    assert 'peak rate: off-peak is half' in upper
    unknown = format_usage({**totals, 'model': 'claude-unreleased'})[1]
    assert unknown.startswith('cost: unknown, no list price recorded')

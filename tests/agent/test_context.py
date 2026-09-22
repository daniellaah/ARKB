"""Context engineering around the real loop: what a long trajectory forgets.

Every run here uses scripted responses and a temporary vault: no model, no
index, no services. What is asserted is what the model is given and what finish
still accepts.
"""

import json

import pytest

from arkb.agent import AgentBudget, AgentObserver, AgentTools, run_agent
from arkb.agent.context import (OFF, ContextPolicy, compact, estimate_tokens, is_compacted,
                                summarize_observation)
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval import ExactRetriever
from tests.agent.helpers import ScriptedModel, reply, tool_call


@pytest.fixture
def vault_tools(linked_vault, engine):
    documents = DocumentAccess(linked_vault, vault_id='v')
    return AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine)


def observer(**budget):
    """An observer whose evidence ruler is four characters per token, like the estimate."""
    return AgentObserver(budget=AgentBudget(**budget) if budget else None,
                         counter=lambda text: len(text) // 4, counter_identity='test:chars/4')


def cite(answer, refs, status='answered'):
    return reply(calls=[tool_call('finish', answer=answer, status=status, evidence_refs=list(refs))])


# --- the long trajectory ----------------------------------------------------

def test_the_oldest_observation_loses_its_body_and_keeps_its_sources(vault_tools):
    """The pairing every transport depends on survives: one answer per call, still JSON."""
    steps = [reply(calls=[tool_call('read', source='Concepts/KV Cache.md')]),
             reply(calls=[tool_call('read', source='Concepts/Attention.md')]),
             lambda messages: cite('Both were read.', [], status='insufficient_evidence')]
    result = run_agent('compare the two notes', client=(model := ScriptedModel(*steps)), tools=vault_tools,
                       model='fake', policy=ContextPolicy(history_tokens=400), observer=observer())
    observations = [m for m in result.state.messages if m['role'] == 'tool']
    assert is_compacted(observations[0]['content']) and not is_compacted(observations[1]['content'])
    summary = json.loads(observations[0]['content'])
    assert summary['sources'] == ['Concepts/KV Cache.md'] and 'Keys and values' not in observations[0]['content']
    assert result.observation['context']['compacted_observations'] == 1
    # The model was sent the compacted conversation, with its calls still answered one for one.
    sent = model.requests[-1]['messages']
    assert [m['role'] for m in sent] == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool']
    assert is_compacted(sent[3]['content'])


def test_a_compacted_observation_stays_citable_because_its_reference_lives_in_the_session(vault_tools):
    """The claim the feature rests on: compaction drops text, never the right to cite it."""
    captured = []

    def remember(messages):
        captured.extend(hit['ref'] for message in messages if message['role'] == 'tool'
                        for hit in [json.loads(message['content'])['result']])
        return reply(calls=[tool_call('read', source='Concepts/Attention.md')])

    steps = [reply(calls=[tool_call('read', source='Concepts/KV Cache.md')]), remember,
             lambda messages: cite('The first note explains the cache.', captured[:1])]
    result = run_agent('why is the cache safe?', client=ScriptedModel(*steps), tools=vault_tools,
                       model='fake', policy=ContextPolicy(history_tokens=400), observer=observer())
    first = [m for m in result.state.messages if m['role'] == 'tool'][0]
    assert is_compacted(first['content']) and captured[0] not in first['content']
    assert result.final.status == 'answered'
    assert [c['ref'] for c in result.final.citations] == captured[:1]
    assert result.final.citations[0]['source'] == 'Concepts/KV Cache.md'
    assert result.observation['reference_resolutions'][-1]['status'] == 'success'


def test_compaction_is_off_at_zero_and_below_the_limit(vault_tools):
    def steps():
        return [reply(calls=[tool_call('read', source='Concepts/KV Cache.md')]),
                lambda messages: cite('Read it.', [], status='insufficient_evidence')]
    for policy in (OFF, ContextPolicy(history_tokens=100000)):
        result = run_agent('read it', client=ScriptedModel(*steps()), tools=vault_tools, model='fake',
                           policy=policy, observer=observer())
        observation = [m for m in result.state.messages if m['role'] == 'tool'][0]
        assert not is_compacted(observation['content']) and 'Keys and values' in observation['content']
        assert 'compacted_observations' not in result.observation['context']


# --- the shared pieces ------------------------------------------------------

def observations(count, *, body='x' * 400):
    return [{'role': 'user', 'content': 'q'},
            *({'role': 'tool', 'tool_name': 'search',
               'content': json.dumps({'results': [{'ref': f'ev_{i}', 'source': f'{i}.md', 'content': body}]})}
              for i in range(count))]


def test_compaction_stops_at_the_limit_and_never_touches_a_summary_twice():
    messages = observations(4)
    halfway = [*messages[:1], *(dict(m, content=summarize_observation(m['content'])) for m in messages[1:3]),
               *messages[3:]]
    kept, count = compact(messages, limit=estimate_tokens(halfway))
    assert count == 2 and kept == halfway
    again, second = compact(kept, limit=1)
    assert second == 2 and all(is_compacted(m['content']) for m in again[1:])
    assert json.loads(again[1]['content'])['sources'] == ['0.md']


def test_the_observations_a_model_has_not_read_are_never_compacted():
    messages = observations(3)
    kept, count = compact(messages, limit=1, protect=2)
    assert count == 1 and kept[2:] == messages[2:] and is_compacted(kept[1]['content'])
    assert compact(messages, limit=1, protect=len(messages))[1] == 0


def test_a_summary_keeps_valid_json_even_for_an_observation_it_cannot_parse():
    assert json.loads(summarize_observation('not json'))['sources'] == []
    assert json.loads(summarize_observation(json.dumps({'status': 'recoverable_error'})))['status'] == 'recoverable_error'


@pytest.mark.parametrize('settings', [{'history_tokens': -1}, {'history_tokens': 'many'}])
def test_a_policy_refuses_settings_it_cannot_honour(settings):
    with pytest.raises(ValueError):
        ContextPolicy(**settings)


def test_the_default_policy_bounds_the_conversation():
    assert ContextPolicy().history_tokens == 24000 and OFF.history_tokens == 0

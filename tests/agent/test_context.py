"""Context engineering around the real loop: a vault map, a small scope, a long trajectory.

Every run here uses scripted responses and a temporary vault: no model, no
index, no services. What is asserted is what the model is given, what the
observer is charged, and what finish still accepts.
"""

import json

import pytest

from arkb.agent import AgentBudget, AgentObserver, AgentTools, run_agent
from arkb.agent.context import (OFF, ContextPolicy, compact, estimate_tokens, is_compacted,
                                map_note, parse_map_notes, scope_estimate, summarize_observation)
from arkb.agent.loop import SYSTEM_INSTRUCTION
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


def system_messages(result):
    return [m['content'] for m in result.state.messages if m['role'] == 'system']


def delivered(result):
    """The evidence the small-scope delivery put in the conversation, as the model sees it."""
    for content in system_messages(result):
        _, _, payload = content.partition('\n\n')
        try:
            value = json.loads(payload)
        except ValueError:
            continue
        if isinstance(value, dict) and 'results' in value:
            return value
    return None


# --- the vault map ----------------------------------------------------------

def test_a_layout_note_opens_the_conversation_as_orientation_and_is_not_evidence(vault_tools, engine):
    policy = ContextPolicy(map_notes=('index.md',), small_scope_tokens=0)
    model = ScriptedModel(cite('Nothing was retrieved.', [], status='insufficient_evidence'))
    result = run_agent('where do concepts live?', client=model, tools=vault_tools, model='fake',
                       policy=policy, observer=(watch := observer()))
    # Orientation arrives before the question, as a system message that says what it is.
    messages = model.requests[0]['messages']
    assert messages[0]['content'] == SYSTEM_INSTRUCTION
    assert messages[1]['role'] == 'system' and messages[-1]['content'] == 'where do concepts live?'
    assert 'index.md' in messages[1]['content'] and 'cannot be cited' in messages[1]['content']
    assert 'The map starts at [[Concepts/KV Cache]]' in messages[1]['content']
    # It is orientation, not evidence: no reference was issued, so nothing became citable.
    assert result.observation['evidence_references'] == {}
    assert result.observation['context'] == {'map_note': 'index.md'}
    assert [e['name'] for e in watch.tools] == ['finish'] and engine.search.call_count == 0


def test_a_bare_filename_finds_the_note_in_its_folder_and_the_first_candidate_wins(vault_tools):
    assert map_note(vault_tools, ('KV Cache.md',))['source'] == 'Archive/KV Cache.md'
    assert map_note(vault_tools, ('Concepts/KV Cache.md', 'index.md'))['source'] == 'Concepts/KV Cache.md'


@pytest.mark.parametrize('candidates', [('nowhere.md',), ('../escape.md', 'Missing/none.md'), ()])
def test_a_missing_layout_note_is_skipped_without_failing_the_run(vault_tools, candidates):
    assert map_note(vault_tools, candidates) is None
    model = ScriptedModel(cite('No knowledge needed.', [], status='insufficient_evidence'))
    result = run_agent('hello', client=model, tools=vault_tools, model='fake',
                       policy=ContextPolicy(map_notes=candidates, small_scope_tokens=0))
    assert result.stop_reason == 'final'
    assert [m['role'] for m in model.requests[0]['messages']] == ['system', 'user']
    assert result.observation['context'] == {}


def test_repeated_and_comma_separated_candidates_become_one_ordered_list():
    assert parse_map_notes(['a.md, b.md', ' c.md ', 'a.md']) == ('a.md', 'b.md', 'c.md')
    assert parse_map_notes(None) == () and parse_map_notes(['  ']) == ()


# --- the small scope --------------------------------------------------------

def test_a_small_scope_is_delivered_whole_instead_of_being_searched(vault_tools, engine):
    sources, estimate = scope_estimate(vault_tools)
    assert len(sources) == 6 and 0 < estimate < 1000
    model = ScriptedModel(lambda messages: cite('All six notes were read.',
                                                [hit['ref'] for hit in delivered_hits(messages)]))
    result = run_agent('what is in this vault?', client=model, tools=vault_tools, model='fake',
                       policy=ContextPolicy(small_scope_tokens=1000), observer=observer())
    # No search ran, and every note arrived as citable evidence in one system message.
    assert engine.search.call_count == 0
    payload = delivered(result)
    assert payload['notes'] == 6 and [hit['source'] for hit in payload['results']] == sources
    assert result.final.status == 'answered'
    assert sorted(c['source'] for c in result.final.citations) == sorted(sources)
    assert all(c['document_revision'] for c in result.final.citations)
    # The run says in its trajectory that it took the bypass, before the first model request.
    assert result.observation['context']['small_scope_bypass'] == {'notes': 6, 'withheld_notes': 0}
    assert [(e['turn'], e['name']) for e in result.observation['tools']] == [(0, 'corpus'), (1, 'finish')]


def delivered_hits(messages):
    payload = None
    for message in messages:
        if message['role'] != 'system':
            continue
        _, _, text = message['content'].partition('\n\n')
        try:
            value = json.loads(text)
        except ValueError:
            continue
        if isinstance(value, dict) and 'results' in value:
            payload = value
    return payload['results'] if payload else []


def test_a_scope_above_the_threshold_keeps_retrieval(vault_tools, engine):
    model = ScriptedModel(reply(calls=[tool_call('search', query='attention')]),
                          cite('Nothing was found.', [], status='insufficient_evidence'))
    result = run_agent('what is attention?', client=model, tools=vault_tools, model='fake',
                       policy=ContextPolicy(small_scope_tokens=10), observer=observer())
    assert engine.search.call_count == 1 and delivered(result) is None
    assert [e['name'] for e in result.observation['tools']] == ['search', 'finish']
    assert result.observation['context']['scope_tokens_estimate'] > 10
    assert 'small_scope_bypass' not in result.observation['context']


def test_the_delivery_is_charged_to_the_evidence_allowance_like_any_observation(vault_tools):
    watch = observer(max_evidence_tokens=60)
    model = ScriptedModel(lambda messages: cite('Partly read.', [h['ref'] for h in delivered_hits(messages)]))
    result = run_agent('what is here?', client=model, tools=vault_tools, model='fake',
                       policy=ContextPolicy(small_scope_tokens=1000), observer=watch)
    payload = delivered(result)
    # A prefix of the scope fits; the rest is withheld and the model is told so.
    assert 0 < payload['notes'] < 6 and payload['withheld_notes'] == 6 - payload['notes']
    assert 'nearly exhausted' in [m for m in system_messages(result) if '"results"' in m][0]
    assert watch.remaining_evidence() >= 0
    assert result.observation['evidence']['delivered_tokens'] <= 60
    # Only what was delivered is citable, and it is.
    assert len(result.final.citations) == payload['notes']


def test_a_scope_that_does_not_fit_at_all_falls_back_to_retrieval_without_charging_anything(vault_tools):
    watch = observer(max_evidence_tokens=5)
    model = ScriptedModel(cite('Nothing was retrieved.', [], status='insufficient_evidence'))
    result = run_agent('what is here?', client=model, tools=vault_tools, model='fake',
                       policy=ContextPolicy(small_scope_tokens=1000), observer=watch)
    assert delivered(result) is None
    assert result.observation['evidence']['delivered_tokens'] == 0
    assert watch.remaining_evidence() == 5 and watch.reason is None
    event = result.observation['tools'][0]
    assert event['name'] == 'corpus' and event['status'] == 'skipped'
    assert event['skip_reason'] == 'evidence_too_large'


def test_a_scope_prefix_delivers_one_folder(vault_tools):
    model = ScriptedModel(lambda messages: cite('Two notes.', [h['ref'] for h in delivered_hits(messages)]))
    result = run_agent('what is in Concepts?', client=model, tools=vault_tools, model='fake',
                       policy=ContextPolicy(small_scope_tokens=1000, scope_prefix='Concepts'),
                       observer=observer())
    assert [hit['source'] for hit in delivered(result)['results']] == ['Concepts/Attention.md', 'Concepts/KV Cache.md']


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


@pytest.mark.parametrize('settings', [{'small_scope_tokens': -1}, {'history_tokens': 'many'},
                                      {'map_notes': 'index.md'}, {'map_notes': (' ',)},
                                      {'scope_prefix': '../outside'}, {'scope_prefix': '/tmp'}])
def test_a_policy_refuses_settings_it_cannot_honour(settings):
    with pytest.raises(ValueError):
        ContextPolicy(**settings)


def test_the_default_policy_only_bounds_the_conversation():
    policy = ContextPolicy()
    assert policy.small_scope_tokens == 0 and policy.map_notes == () and policy.history_tokens == 24000
    assert OFF.history_tokens == 0

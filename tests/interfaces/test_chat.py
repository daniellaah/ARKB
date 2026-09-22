"""The chat session as a function over input lines; no terminal, no model, no services."""

import json

import pytest

from arkb.agent.state import AgentFinal, AgentResult, AgentState, ObservedAgentResult
from arkb.interfaces.chat import (carry_history, converse, estimate_tokens, is_compacted,
                                  summarize_observation)


def observation(*sources, text='a long body ' * 20):
    return json.dumps({'status': 'success', 'query': 'q',
                       'results': [{'ref': f'ev_{s}', 'source': s, 'content': text} for s in sources]})


def answered(query, history, answer='an answer', *, calls=()):
    """What one agent run leaves behind: the instruction, the replayed history, this turn.

    The observation records only this turn's calls, as a per-turn observer does.
    """
    messages = [{'role': 'system', 'content': 'instructions'}, *history, {'role': 'user', 'content': query}]
    for name, arguments in calls:
        messages += [{'role': 'assistant', 'tool_calls': [{'function': {'name': name, 'arguments': arguments}}]},
                     {'role': 'tool', 'tool_name': name, 'content': observation('a.md')}]
    messages.append({'role': 'assistant', 'content': answer})
    events = [{'turn': 1, 'name': name, 'arguments': arguments, 'conversation_result': {'status': 'success'}}
              for name, arguments in calls]
    return ObservedAgentResult(answer, 'final', AgentState(messages, 1), {'tools': events},
                               final=AgentFinal(answer, 'answered', [], 'finish'))


class Session:
    """Records the queries and the history each turn was given."""

    def __init__(self, reply=lambda query, history: answered(query, history)):
        self.reply, self.turns, self.resets = reply, [], 0

    def run(self, query, history):
        self.turns.append((query, list(history)))
        return self.reply(query, history)

    def reset(self):
        self.resets += 1


def test_a_blank_line_ends_the_session_before_anything_runs():
    session = Session()
    assert list(converse(iter(['', 'never asked']), run=session.run)) == []
    assert session.turns == []


def test_each_turn_replays_the_previous_conversation_without_its_operator_instructions():
    session = Session()
    lines = list(converse(iter(['first question', 'and the follow-up?', '']), run=session.run))
    assert lines == ['an answer', 'an answer']
    first, second = session.turns
    assert first == ('first question', [])
    assert second[0] == 'and the follow-up?'
    # The instruction belongs to the run that ended; the next run supplies its own.
    assert all(m['role'] != 'system' for m in second[1])
    assert [m['content'] for m in second[1]] == ['first question', 'an answer']


def test_reset_drops_the_conversation_and_whatever_the_caller_keeps_beside_it():
    session = Session()
    lines = list(converse(iter(['first', '/reset', 'second', '']), run=session.run, reset=session.reset))
    assert lines[1].startswith('Session cleared')
    assert session.resets == 1
    assert [history for _, history in session.turns] == [[], []]


def test_trace_toggles_the_trajectory_and_the_usage_summary():
    session = Session(lambda query, history: answered(query, history, calls=[('match', {'query': 'foo'})]))
    lines = list(converse(iter(['q1', '/trace', 'q2', '']), run=session.run,
                          usage=lambda: ['usage: 1 request(s)']))
    assert lines[0] == 'an answer' and lines[1] == 'Trace on.'
    # The trajectory covers the turn that just ran, not every call the session ever made.
    assert lines[2:] == ['[1] match', 'query: "foo"', '', '[2] final', 'an answer', 'usage: 1 request(s)']
    # Starting with trace on and switching it off is the same toggle.
    assert list(converse(iter(['/trace', 'q', '']), run=Session().run, trace=True)) == ['Trace off.', 'an answer']


def test_a_turn_without_a_final_response_is_reported_rather_than_invented():
    session = Session(lambda query, history: AgentResult(
        None, 'budget', AgentState([{'role': 'user', 'content': query}], 1),
        final=AgentFinal(None, 'error', [], 'max_elapsed_ms')))
    assert list(converse(iter(['q', '']), run=session.run)) == ['No final response: max_elapsed_ms.']


def test_the_oldest_observations_lose_their_bodies_first_and_keep_their_sources():
    history = [{'role': 'user', 'content': 'q'},
               {'role': 'tool', 'tool_name': 'search', 'content': observation('a.md', 'b.md')},
               {'role': 'tool', 'tool_name': 'read', 'content': observation('c.md')},
               {'role': 'assistant', 'content': 'answer'}]
    full = estimate_tokens(history)
    kept = carry_history(history, limit=full - 1)
    assert estimate_tokens(kept) < full
    first, second = json.loads(kept[1]['content']), json.loads(kept[2]['content'])
    assert first['sources'] == ['a.md', 'b.md'] and first['compacted'] is True and 'results' not in first
    # Compaction stops as soon as the conversation fits; the newest observation keeps its text.
    assert second == json.loads(history[2]['content'])
    assert kept[0] == history[0] and kept[3] == history[3]
    assert carry_history(history, limit=0) == history
    assert carry_history(history, limit=full * 10) == history


def test_compacting_twice_does_not_lose_the_sources_the_first_pass_kept():
    once = summarize_observation(observation('a.md', 'b.md'))
    assert is_compacted(once) and not is_compacted(observation('a.md'))
    history = [{'role': 'tool', 'tool_name': 'search', 'content': once}]
    assert carry_history(history, limit=1) == history
    assert json.loads(summarize_observation('not json'))['sources'] == []


@pytest.mark.parametrize('command', ['/reset', '/trace'])
def test_commands_do_not_reach_the_agent(command):
    session = Session()
    list(converse(iter([command, '']), run=session.run, reset=session.reset))
    assert session.turns == []

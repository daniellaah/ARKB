"""Follow-up turns over the real loop: carried history, per-turn budgets, cross-turn references."""

import json

from arkb.agent import AgentBudget, AgentObserver, run_agent
from arkb.agent.loop import SYSTEM_INSTRUCTION
from arkb.agent.session import ToolSession
from arkb.interfaces.chat import carry_history
from tests.agent.helpers import ScriptedModel, reply, tool_call

BUDGET = AgentBudget(max_tool_calls=12, max_query_calls=10, max_read_calls=6)


def refs_in(messages):
    """Every evidence reference the conversation has seen, in order."""
    found = []
    for message in messages:
        if message['role'] != 'tool':
            continue
        result = json.loads(message['content'])
        hits = [result['result']] if 'result' in result else result.get('results', [])
        found += [hit['ref'] for hit in hits if isinstance(hit, dict) and hit.get('ref') not in found]
    return found


def cite(answer, refs, status='answered'):
    return reply(calls=[tool_call('finish', answer=answer, status=status, evidence_refs=list(refs))])


def take_turn(query, *steps, tools, session, history=(), budget=BUDGET):
    """One complete agent run with its own observer, sharing the caller's tool session."""
    model = ScriptedModel(*steps)
    result = run_agent(query, client=model, tools=tools, model='fake', history=history,
                       session_factory=lambda _tools: session, observer=AgentObserver(budget=budget))
    return result, model


def first_turn(tools, session):
    return take_turn('which note mentions café?',
                     reply(calls=[tool_call('match', query='café')]),
                     lambda messages: cite('Alpha does.', refs_in(messages)[:1]),
                     tools=tools, session=session)


def test_a_follow_up_sees_the_earlier_turn_and_can_still_cite_its_evidence(tools):
    session = ToolSession(tools)
    first, _ = first_turn(tools, session)
    assert first.final.status == 'answered' and [c['source'] for c in first.final.citations] == ['a.md']
    earlier = first.final.citations[0]['ref']

    history = carry_history(first.state.messages)
    assert all(message['role'] != 'system' for message in history)
    second, model = take_turn('and what else is in it?', cite('A detail section.', [earlier]),
                              tools=tools, session=session, history=history)
    # The reference collected in the first turn resolves in the second: same session, same revision.
    assert second.final.status == 'answered' and second.final.citations[0]['ref'] == earlier
    replayed = model.requests[0]['messages']
    assert replayed[0] == {'role': 'system', 'content': SYSTEM_INSTRUCTION}
    assert replayed[1]['content'] == 'which note mentions café?'
    assert replayed[-1] == {'role': 'user', 'content': 'and what else is in it?'}
    # Turn two spent no tool call to reuse evidence turn one had already fetched.
    assert [call.name for call in second.trace.tool_calls] == ['finish']
    assert second.state.turn == 1


def test_a_new_session_refuses_references_from_the_conversation_it_replaced(tools):
    first, _ = first_turn(tools, ToolSession(tools))
    earlier = first.final.citations[0]['ref']
    # /reset keeps nothing: a fresh session has never issued that reference.
    result, _ = take_turn('and what else is in it?', cite('A detail section.', [earlier]),
                          cite('I need to look again.', [], status='insufficient_evidence'),
                          tools=tools, session=ToolSession(tools),
                          history=carry_history(first.state.messages))
    rejected = result.trace.tool_calls[0].result
    assert rejected['error']['code'] == 'invalid_reference'
    assert result.final.status == 'insufficient_evidence'


def test_each_turn_gets_its_own_allowance(tools):
    """A budget spent in one turn leaves no debt for the next."""
    session, budget = ToolSession(tools), AgentBudget(max_tool_calls=2, max_query_calls=2)
    spend = (reply(calls=[tool_call('match', query='café')]), reply(calls=[tool_call('match', query='foo')]),
             lambda messages: cite('Alpha.', refs_in(messages)[:1]))
    first, _ = take_turn('q1', *spend, tools=tools, session=session, budget=budget)
    second, _ = take_turn('q2', *spend, tools=tools, session=session, budget=budget,
                          history=carry_history(first.state.messages))
    assert [sum(call.name == 'match' for call in result.trace.tool_calls) for result in (first, second)] == [2, 2]
    assert first.final.status == second.final.status == 'answered'

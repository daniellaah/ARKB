from dataclasses import asdict

import pytest

from arkb.agent.loop import run_agent, _SEARCH_STALLED_INSTRUCTION
from tests.agent.helpers import ScriptedModel, reply, tool_call, complete
from tests.agent.test_observation import normalized, observer


def test_ablation_changes_only_reminder_and_preserves_default(tools):
    steps = [reply(calls=[tool_call('search', query='q')]) for _ in range(3)] + [complete('done')]
    models = [ScriptedModel(*steps) for _ in range(3)]
    results = [run_agent('q', client=m, tools=tools, model='fake', observer=observer(), **kw)
               for m, kw in zip(models, [{}, {'search_stall_reminder': True}, {'search_stall_reminder': False}])]
    assert normalized(models[0].requests) == normalized(models[1].requests)
    assert normalized(asdict(results[0].trace)) == normalized(asdict(results[1].trace)) == normalized(asdict(results[2].trace))
    assert models[0].requests[:3] == models[2].requests[:3]
    on, off = models[0].requests[-1], models[2].requests[-1]
    assert on['messages'][-1]['content'] == _SEARCH_STALLED_INSTRUCTION
    assert {**on, 'messages': on['messages'][:-1]} == off


@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_ablation_requires_boolean(tools, value):
    m = ScriptedModel(complete('done'))
    with pytest.raises(ValueError, match='search_stall_reminder'):
        run_agent('q', client=m, tools=tools, model='fake', search_stall_reminder=value)
    assert not m.requests

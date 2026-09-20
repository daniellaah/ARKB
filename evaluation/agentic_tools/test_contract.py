"""Synthetic boundary checks; no services, model calls, or benchmark answers."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
from ollama import ChatResponse

from arkb.agent.loop import run_agent, SYSTEM_INSTRUCTION, _FINAL_INSTRUCTION
from arkb.agent.tools import tool_definitions
from arkb.evaluation.multihop import canonical_prediction
from arkb.agent.state import AgentFinal
from .contract import (ARMS, ARM_BY_ID, OPTIONS, THINK, RestrictedTools, RestrictedSession,
                       controlled_agent, fixed_rag, new_observer, pack_prefix, evidence_sets, rendered_prompt)
from .selection import build_selection, schedule, attempt_key


class Capabilities:
    _rerank = False
    _mode = 'semantic'
    _engine = SimpleNamespace(candidate_k=20, rerank_candidates=20)

    def __init__(self, texts=('alpha beta', 'gamma delta')):
        self.calls = []
        self.hits = [{'document_id': str(i), 'source': f'{i}.md', 'title': '', 'content': text,
                      'document_revision': 'rev-' + str(i), 'chunk_id': str(i), 'section_id': None,
                      'start_char': 0, 'end_char': len(text)} for i, text in enumerate(texts)]

    def search(self, query, **kwargs):
        self.calls.append(('search', query, kwargs))
        return {'query': query, 'results': deepcopy(self.hits), 'index_id': 'frozen'}

    def match(self, query, **kwargs):
        self.calls.append(('match', query, kwargs))
        return {'query': query, 'results': deepcopy(self.hits)}

    def read(self, document_id=None, *, source=None):
        hit = next(h for h in self.hits if h['source'] == source)
        return {'result': deepcopy(hit)}


def response(content='', calls=()):
    return ChatResponse(model='qwen3.5:4b', done=True, done_reason='stop', prompt_eval_count=25, eval_count=12,
                        message={'role': 'assistant', 'content': content,
                                 'tool_calls': [{'function': {'name': n, 'arguments': a}} for n, a in calls]})


class Client:
    def __init__(self, actions, *, options=OPTIONS, think=THINK):
        self.actions, self.requests = iter(actions), []
        self.options, self.think = options, think

    def chat(self, **request):
        self.requests.append(deepcopy(request))
        assert request['options'] == self.options and request['think'] in (False, self.think)
        action = next(self.actions)
        return action(request) if callable(action) else action


def obs(counter=None, **configuration):
    return new_observer(counter or (lambda text: len(text.split())), 'fixture-words-v1', **configuration)


@pytest.mark.parametrize('arm', ARMS)
def test_schema_and_boundary_reject_mode_escape(arm):
    base = Capabilities()
    tools = RestrictedTools(base, arm)
    s = RestrictedSession(tools)
    expected = {'finish', 'read'} | ({'search'} if arm.modes else set()) | ({'match'} if arm.match else set())
    assert {d['name'] for d in s.definitions} == expected
    for mode in ['bm25', 'semantic', 'hybrid', 'lexical', 'not-a-mode']:
        n = len(base.calls)
        r = s.invoke('search', {'query': 'x', 'mode': mode})
        assert (r['status'] == 'success') is (mode in arm.modes)
        if mode not in arm.modes:
            assert len(base.calls) == n
    for args in [{'query': 'x'}, {'query': 'x', 'mode': None}]:
        r = s.invoke('search', args)
        if arm.modes:
            assert r['status'] == 'success' and base.calls[-1][2]['mode'] == arm.default_mode
        else:
            assert r['status'] == 'recoverable_error'
    assert (s.invoke('match', {'query': 'x'})['status'] == 'success') is arm.match
    for bad in [{'document_id': '0'}, {'source': '../0.md'}, {'source': '0.md', 'mode': 'hybrid'},
                {'ref': 'fake'}, {'source': '0.md', 'start_char': 0}]:
        assert s.invoke('read', bad)['status'] == 'recoverable_error'
    assert s.invoke('read', {'source': '0.md'})['status'] == 'success'


def test_schema_does_not_mutate_product_and_hidden_prompt_hints_removed():
    before = tool_definitions(('bm25', 'semantic', 'hybrid'), default_mode='semantic')
    for arm in ARMS:
        RestrictedTools(Capabilities(), arm).tool_definitions()
    assert before == tool_definitions(('bm25', 'semantic', 'hybrid'), default_mode='semantic')
    assert 'search' not in rendered_prompt(ARM_BY_ID['A-M'])
    assert 'match' not in rendered_prompt(ARM_BY_ID['A-S'])


def test_multicall_escape_recorded_without_executing_forbidden_tools():
    base = Capabilities()
    client = Client([response(calls=[('match', {'query': 'x'}), ('search', {'query': 'x', 'mode': 'hybrid'}),
                                    ('search', {'query': 'x'})]), response('proposal'),
                     response(json.dumps({'answer': 'Insufficient evidence', 'status': 'insufficient_evidence', 'evidence_refs': []}))])
    result = controlled_agent('q', tools=base, arm=ARM_BY_ID['A-S'], client=client, observer=obs())
    assert len(base.calls) == 1 and base.calls[0][2]['mode'] == 'semantic'
    assert len(result.observation['tools']) == 3
    assert [e['status'] for e in result.observation['tools']] == ['recoverable_error', 'recoverable_error', 'success']
    assert run_agent.__globals__['SYSTEM_INSTRUCTION'] == SYSTEM_INSTRUCTION
    assert result.state.messages[0]['content'] == rendered_prompt(ARM_BY_ID['A-S'])
    assert all(m['request']['options'] == OPTIONS for m in result.observation['models'])


def test_rank_prefix_stops_at_first_unfit_and_charges_duplicates():
    hits = [{'ref': str(i), 'content': s} for i, s in enumerate(['a a', 'a a', 'x x x', 'z'])]
    packed, decisions = pack_prefix(hits, lambda x: len(x.split()), 5)
    assert [x['ref'] for x in packed] == ['0', '1']
    assert [x['reason'] for x in decisions] == ['included', 'included', 'evidence_ceiling', 'after_first_unfit']


def test_fixed_rag_reasons_then_formats_with_packed_reference_visibility():
    base = Capabilities(('a', 'x ' * 8001, 'z'))
    def reason(request):
        evidence = json.loads(request['messages'][-1]['content'].split('\n', 1)[1])
        assert len(evidence['results']) == 1
        assert 'tools' not in request and 'format' not in request
        draft = response('The answer is a, from ' + evidence['results'][0]['ref'])
        draft.message.thinking = 'private reasoning'
        return draft
    def answer(request):
        assert 'format' in request and request['think'] is False
        assert request['messages'][-1]['content'] == _FINAL_INSTRUCTION
        assert request['messages'][-2]['role'] == 'assistant' and 'thinking' not in request['messages'][-2]
        ref = request['messages'][-2]['content'].rsplit(' ', 1)[1]
        return response(json.dumps({'answer': 'a', 'status': 'answered', 'evidence_refs': [ref]}))
    client = Client([reason, answer])
    result = fixed_rag('original question', tools=base, arm=ARM_BY_ID['F-H'], client=client, observer=obs())
    assert result.final.status == 'answered' and len(client.requests) == 2
    assert [m['phase'] for m in result.observation['models']] == ['fixed_reasoning', 'fixed_generation']
    assert result.observation['models'][0]['response']['message']['thinking'] == 'private reasoning'
    assert base.calls == [('search', 'original question', {'mode': 'hybrid', 'limit': 20})]
    assert result.observation['evidence']['returned_tokens'] == 8003
    assert result.observation['evidence']['delivered_tokens'] == 1
    assert evidence_sets(result.observation) == {'returned': ['0.md', '1.md', '2.md'],
                                                'delivered': ['0.md'], 'submitted': ['0.md']}


def test_agent_whole_oversize_observation_is_withheld():
    final = response(json.dumps({'answer': 'No evidence', 'status': 'insufficient_evidence', 'evidence_refs': []}))
    client = Client([response(calls=[('search', {'query': 'q'})]), final, final])
    result = controlled_agent('q', tools=Capabilities(('x ' * 8001,)), arm=ARM_BY_ID['A-S'], client=client, observer=obs())
    assert result.final.status == 'insufficient_evidence'
    # The oversized hit is withheld; the untouched allowance keeps collection open.
    assert result.observation['budget_stop_reason'] is None
    assert result.observation['tools'][0]['withheld_hits'] == 1
    assert evidence_sets(result.observation) == {'returned': ['0.md'], 'delivered': [], 'submitted': []}
    assert result.observation['evidence']['delivered_tokens'] == 0


def test_agent_search_results_are_delivered_as_the_fitting_prefix():
    final = response(json.dumps({'answer': 'x', 'status': 'answered', 'evidence_refs': []}))
    client = Client([response(calls=[('search', {'query': 'q'})]), final, final])
    texts = ('x ' * 3000, 'y ' * 3000, 'z ' * 3000)
    result = controlled_agent('q', tools=Capabilities(texts), arm=ARM_BY_ID['A-S'], client=client, observer=obs())
    event = result.observation['tools'][0]
    assert event['delivered_to_conversation'] and event['withheld_hits'] == 1 and len(event['delivered_refs']) == 2
    assert evidence_sets(result.observation) == {'returned': ['0.md', '1.md', '2.md'], 'delivered': ['0.md', '1.md'], 'submitted': ['0.md', '1.md']}
    conversation = json.loads(result.state.messages[3]['content'])
    assert conversation['withheld_results'] == 1 and [h['source'] for h in conversation['results']] == ['0.md', '1.md']
    assert result.observation['evidence']['delivered_tokens'] == 6000


def test_reserved_finalization_eighth_request_and_no_product_global_change():
    client = Client([response(calls=[('match', {'query': ''})]) for _ in range(7)] +
                    [response(json.dumps({'answer': 'No evidence', 'status': 'insufficient_evidence', 'evidence_refs': []}))])
    result = controlled_agent('q', tools=Capabilities(), arm=ARM_BY_ID['A-M'], client=client, observer=obs())
    assert len(client.requests) == 8 and 'format' in client.requests[-1] and 'tools' not in client.requests[-1]
    assert result.final.termination_reason == 'max_turns'


@pytest.mark.parametrize('status,expected', [('answered', True), ('partial', False), ('insufficient_evidence', False), ('error', None)])
def test_musique_mapping_never_parses_prose_or_credits_failure(status, expected):
    final = AgentFinal('unchanged prose', status, [], 'fixture')
    p, _ = canonical_prediction(final, {})
    assert p['predicted_answerable'] is expected
    assert p['predicted_answer'] == ('' if status == 'error' else 'unchanged prose')


def test_selection_and_schedule_are_complete_pair_preserving_and_order_independent():
    populations = {k: [str(i) for i in range(n)] for k, n in [('browsecomp-plus', 830), ('fiqa', 648), ('nfcorpus', 323)]}
    pairs = [f'{h}hop{1 if h > 2 else ""}__{i}' for h, n in [(2, 34), (3, 33), (4, 33)] for i in range(n)]
    chosen = build_selection(populations, pairs)
    assert chosen == build_selection({k: list(reversed(v)) for k, v in populations.items()}, reversed(pairs))
    pilot, core = schedule(chosen, 'pilot'), schedule(chosen, 'core')
    assert len(pilot) == 98 and len(core) == 5460
    assert len({attempt_key('frozen', r) for r in core}) == len(core)
    assert not {(r['dataset'], r['id']) for r in core} & {(r['dataset'], r['id']) for r in pilot}
    for rows in [pilot, core]:
        mu = [r for r in rows if r['dataset'] == 'musique']
        for a, b in zip(mu[::2], mu[1::2]):
            assert a['id'] == b['id'] and a['arm'] == b['arm'] and a['repetition'] == b['repetition']
            assert [a['variant'], b['variant']] == ['v0', 'v1']


def final_json(answer='No evidence', status='insufficient_evidence'):
    return response(json.dumps({'answer': answer, 'status': status, 'evidence_refs': []}))


def test_answer_format_sentence_reaches_every_arm():
    for arm in ARMS:
        assert 'shortest complete answer on its own first line' in rendered_prompt(arm)


def test_agent_adapter_records_supplied_model_and_thinks_only_while_collecting():
    configuration = dict(model='qwen3.5:9b', think=True, options={'temperature': 0, 'num_ctx': 16384, 'num_predict': 2048})
    client = Client([response(calls=[('search', {'query': 'q'})]), final_json(), final_json()],
                    options=configuration['options'], think=True)
    result = controlled_agent('q', tools=Capabilities(), arm=ARM_BY_ID['A-S'], client=client, observer=obs(**configuration),
                              **configuration)
    assert result.final.status == 'insufficient_evidence'
    models = result.observation['models']
    assert [m['request']['model'] for m in models] == ['qwen3.5:9b'] * len(models)
    assert all(m['request']['options'] == configuration['options'] for m in models)
    assert [m['request']['think'] for m in models] == [True, True, False]
    assert [m['phase'] for m in models] == ['tools', 'tools', 'finalization']
    assert [r['think'] for r in client.requests] == [True, True, False]


def test_fixed_adapter_thinks_while_reasoning_and_not_while_formatting():
    configuration = dict(model='qwen3.5:9b', think=True, options={'temperature': 0, 'num_ctx': 16384, 'num_predict': 2048})
    client = Client([response('draft'), final_json()], options=configuration['options'], think=True)
    result = fixed_rag('q', tools=Capabilities(), arm=ARM_BY_ID['F-S'], client=client, observer=obs(**configuration), **configuration)
    assert result.final.status == 'insufficient_evidence'
    requests = [m['request'] for m in result.observation['models']]
    assert [r['model'] for r in requests] == ['qwen3.5:9b'] * 2 and all(r['options'] == configuration['options'] for r in requests)
    assert [r['think'] for r in requests] == [True, False] and [r['think'] for r in client.requests] == [True, False]
    assert 'format' not in requests[0] and 'format' in requests[1]


def test_truncated_fixed_reasoning_is_a_failure_without_a_formatting_request():
    truncated = ChatResponse(model='qwen3.5:4b', done=True, done_reason='length', message={'role': 'assistant', 'content': ''})
    client = Client([truncated, final_json()])
    result = fixed_rag('q', tools=Capabilities(), arm=ARM_BY_ID['F-S'], client=client, observer=obs())
    assert result.final.status == 'error' and result.final.error['stage'] == 'model_protocol'
    assert len(client.requests) == 1 and len(result.observation['models']) == 1


def test_defaults_keep_the_registered_contract_and_mismatched_observer_is_rejected():
    client = Client([response('draft'), final_json()])
    result = fixed_rag('q', tools=Capabilities(), arm=ARM_BY_ID['F-S'], client=client, observer=obs())
    for request in (m['request'] for m in result.observation['models']):
        assert request['model'] == 'qwen3.5:4b' and request['think'] is False and request['options'] == OPTIONS
    with pytest.raises(ValueError, match='Observer configuration'):
        fixed_rag('q', tools=Capabilities(), arm=ARM_BY_ID['F-S'], client=Client([final_json()]), observer=obs(), think=True)
    with pytest.raises(ValueError, match='Observer configuration'):
        controlled_agent('q', tools=Capabilities(), arm=ARM_BY_ID['A-S'], client=Client([final_json()]),
                         observer=obs(model='qwen3.5:9b'))

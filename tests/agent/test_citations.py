"""The citation rule in the prompt, and the check after the answer; no real model."""
import json

import pytest

from arkb.agent import AgentTools, run_agent
from arkb.agent import citations
from arkb.agent.citations import DISCIPLINE, VERIFY_INSTRUCTION, CitationPolicy
from arkb.agent.loop import _FINAL_INSTRUCTION
from arkb.agent.observation import AgentBudget, AgentObserver
from arkb.retrieval import BM25Retriever, ExactRetriever, RetrievalEngine
from tests.agent.helpers import ScriptedModel, complete, reply, tool_call


@pytest.fixture
def bm25_tools(documents):
    engine = RetrievalEngine(bm25=BM25Retriever(list(documents.records()), index_id='test'))
    return AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine, mode='bm25')


class CheckedModel(ScriptedModel):
    """A scripted run whose verification requests are answered from `verdicts`.

    Verification requests are answered out of band, not from the script, so a
    test states the trajectory and the verdicts separately. A verdict may be a
    bool, raw response text, or an exception the transport raises.
    """

    def __init__(self, *steps, verdicts=()):
        super().__init__(*steps)
        self.verdicts = list(verdicts)
        self.checks = []

    def chat(self, **request):
        if request['messages'][0]['content'] != VERIFY_INSTRUCTION:
            return super().chat(**request)
        self.checks.append(request)
        verdict = self.verdicts.pop(0) if self.verdicts else True
        if isinstance(verdict, Exception):
            raise verdict
        return reply(verdict if isinstance(verdict, str) else json.dumps({'supported': verdict}))


def two_citations(**policy):
    """One run that reads two notes and cites both, under the given citation policy."""
    def read_both(messages):
        return reply(calls=[tool_call('read', source='b.md')])
    steps = (reply(calls=[tool_call('read', source='a.md')]), read_both, complete('Both notes'))
    return steps, CitationPolicy(**policy)


def run(tools, model, policy, **kwargs):
    return run_agent('question', client=model, tools=tools, model='fake', citation_policy=policy, **kwargs)


# --- the rule in the prompt --------------------------------------------------

def test_discipline_is_off_by_request_and_leaves_the_prompt_untouched(bm25_tools):
    model = ScriptedModel(reply(calls=[tool_call('read', source='a.md')]), complete('answer'))
    run(bm25_tools, model, CitationPolicy())
    finish = [d['function'] for r in model.requests for d in r.get('tools') or ()
              if d['function']['name'] == 'finish']
    assert finish and all(DISCIPLINE not in d['description'] for d in finish)
    assert all(DISCIPLINE not in m['content'] for r in model.requests for m in r['messages']
               if m['role'] == 'system')


def test_discipline_states_the_rule_where_finish_is_chosen_and_where_it_is_forced(bm25_tools):
    calls = [tool_call('read', source='a.md')]
    model = ScriptedModel(reply(calls=calls), reply('No more tools'), complete('answer', constrained=True))
    result = run(bm25_tools, model, CitationPolicy(discipline=True))
    assert result.stop_reason == 'final'
    finish = [d['function'] for d in model.requests[0]['tools'] if d['function']['name'] == 'finish']
    assert finish[0]['description'].endswith(DISCIPLINE)
    closing = [m['content'] for m in model.requests[-1]['messages'] if m['role'] == 'system']
    assert closing[-1] == f'{_FINAL_INSTRUCTION} {DISCIPLINE}'
    # The advertised schema is untouched: only the description carries the rule.
    advertised = {d['name']: d['parameters'] for d in bm25_tools.tool_definitions()}
    assert {d['function']['name']: d['function']['parameters'] for d in model.requests[0]['tools']} == advertised


# --- the check after the answer ----------------------------------------------

def test_verification_is_off_by_default_and_spends_no_extra_request(bm25_tools):
    steps, _ = two_citations()
    model = CheckedModel(*steps)
    result = run(bm25_tools, model, CitationPolicy())
    assert [c['source'] for c in result.final.citations] == ['a.md', 'b.md']
    assert model.checks == [] and len(model.requests) == 3


def test_verification_drops_the_citation_the_model_cannot_tie_to_the_answer(bm25_tools):
    steps, policy = two_citations(verify=True)
    model = CheckedModel(*steps, verdicts=[True, False])
    observer = AgentObserver()
    result = run(bm25_tools, model, policy, observer=observer)
    assert result.final.status == 'answered'
    assert [c['source'] for c in result.final.citations] == ['a.md']
    # One request per citation, each carrying the answer and one note, and no tools.
    assert len(model.checks) == 2
    assert all(len(c['messages']) == 2 and 'tools' not in c and c['think'] is False for c in model.checks)
    assert [c['messages'][1]['content'].count('Both notes') for c in model.checks] == [1, 1]
    assert 'a.md' in model.checks[0]['messages'][1]['content']
    assert 'b.md' in model.checks[1]['messages'][1]['content']
    report = result.observation
    assert report['citations'] == {'checked': 2, 'kept': 1, 'dropped': ['b.md']}
    # The extra requests are on the books: recorded as model requests with their own phase.
    assert [m['phase'] for m in report['models']] == ['tools', 'tools', 'tools',
                                                      'citation_verification', 'citation_verification']


def test_dropping_every_citation_downgrades_the_status(bm25_tools):
    steps, policy = two_citations(verify=True)
    model = CheckedModel(*steps, verdicts=[False, False])
    result = run(bm25_tools, model, policy)
    assert result.final.status == 'insufficient_evidence' and result.final.citations == []
    assert result.response == 'Both notes' and result.stop_reason == 'final'
    assert result.observation['citations'] == {'checked': 2, 'kept': 0, 'dropped': ['a.md', 'b.md'],
                                               'status': 'answered -> insufficient_evidence'}


@pytest.mark.parametrize('verdict', ['not json', json.dumps({'supported': 'no'}), ValueError('transport down')])
def test_a_check_that_cannot_decide_keeps_its_citation(bm25_tools, verdict):
    steps, policy = two_citations(verify=True)
    model = CheckedModel(*steps, verdicts=[verdict, False])
    result = run(bm25_tools, model, policy)
    assert [c['source'] for c in result.final.citations] == ['a.md']
    assert result.final.status == 'answered'


def test_a_final_without_citations_is_not_checked_or_downgraded(tools):
    model = CheckedModel(complete('Hello!', status='answered', constrained=True))
    result = run(tools, model, CitationPolicy(verify=True), max_turns=1)
    assert result.final.status == 'answered' and result.final.citations == []
    assert model.checks == [] and result.observation['citations'] == {}


def test_verification_stops_at_the_run_deadline_and_keeps_what_it_did_not_check(bm25_tools):
    steps, policy = two_citations(verify=True)
    model = CheckedModel(*steps, verdicts=[False, False])
    # The clock passes the deadline as soon as the first citation has been checked.
    observer = AgentObserver(budget=AgentBudget(max_elapsed_ms=1000),
                             clock=lambda: 10.0 if model.checks else 0.0)
    result = run(bm25_tools, model, policy, observer=observer)
    assert len(model.checks) == 1
    assert [c['source'] for c in result.final.citations] == ['b.md']
    assert result.observation['citations'] == {'checked': 2, 'kept': 1, 'dropped': ['a.md']}


# --- the request itself ------------------------------------------------------

def test_a_long_note_is_cut_before_it_is_checked():
    citation = {'source': 'a.md', 'title': 'Alpha', 'content': 'x' * (citations.VERIFY_MAX_CHARS + 10)}
    request = citations.verification_request('fake', answer='answer', citation=citation)
    body = request['messages'][1]['content']
    assert citations.CUT_NOTE in body and len(body) < citations.VERIFY_MAX_CHARS + 500
    assert request['format'] == citations.VERIFY_SCHEMA and request['options']['temperature'] == 0


@pytest.mark.parametrize('response,expected', [
    (reply(json.dumps({'supported': True})), True),
    (reply(json.dumps({'supported': False})), False),
    (reply(json.dumps({'supported': 1})), None),
    (reply('yes'), None),
    (reply(None), None),
    (reply(json.dumps({'supported': True}), done_reason='length'), None)])
def test_only_a_boolean_verdict_counts(response, expected):
    assert citations.supported(response) is expected


def test_the_policy_rejects_anything_but_booleans():
    with pytest.raises(ValueError, match='booleans'):
        CitationPolicy(verify='yes')
    with pytest.raises(ValueError, match='CitationPolicy'):
        run_agent('q', client=None, tools=None, model='fake', citation_policy='on')

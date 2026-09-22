"""The list tool: browse filenames, titles and headings without producing evidence."""
import pytest

from arkb.agent.observation import AgentBudget, AgentObserver
from arkb.agent.session import ToolSession


def test_list_summarizes_notes_with_headings_and_filters_by_name(documents):
    listing = documents.list()
    assert [n['source'] for n in listing['notes']] == ['a.md', 'b.md', 'empty.md']
    assert listing['total'] == 3 and listing['truncated'] is False
    alpha = listing['notes'][0]
    assert alpha['title'] == 'Alpha' and alpha['headings'] == ['## Detail'] and alpha['chars'] == len('éø café foo() fooX\n\n## Detail\nrare idea')
    assert [n['source'] for n in documents.list('A')['notes']] == ['a.md']
    assert [n['source'] for n in documents.list('*.md', limit=2)['notes']] == ['a.md', 'b.md']
    assert documents.list('*.md', limit=2)['truncated'] is True and documents.list('zzz')['notes'] == []
    with pytest.raises(ValueError):
        documents.list(' ')


def test_session_list_is_validated_counts_as_retrieval_and_carries_no_evidence(tools):
    session = ToolSession(tools)
    assert 'list' in session._schemas and session.retrieval_attempted is False
    result = session.invoke('list', {'pattern': 'b'})
    assert result['status'] == 'success' and [n['source'] for n in result['notes']] == ['b.md']
    assert session.retrieval_attempted is True and session.references == {}
    assert session.invoke('list', {'limit': 0})['error']['code'] == 'invalid_arguments'
    assert session.invoke('list', {'pattern': 'b', 'extra': 1})['error']['code'] == 'invalid_arguments'
    # An answer after only listing still needs cited evidence.
    assert session.invoke('finish', {'answer': 'b.md is about beta', 'status': 'answered', 'evidence_refs': []})['error']['code'] == 'invalid_citation'


def test_observer_delivers_a_listing_without_charging_the_evidence_allowance(tools):
    from types import SimpleNamespace
    observer = AgentObserver(budget=AgentBudget(max_tool_calls=2, max_query_calls=1, max_read_calls=1, max_evidence_tokens=10, max_elapsed_ms=None),
                             counter=lambda text: len(text.split()), counter_identity='words')
    observer.start()
    session = ToolSession(tools)
    call = SimpleNamespace(function=SimpleNamespace(name='list', arguments={}))
    event = observer.request_tools(0, [call])[0]
    assert observer.permit_tool(event)
    observer.start_tool(event)
    assert observer.end_tool(event, session.invoke('list', {}), references=session.references)
    assert event['delivered_to_conversation'] and event['delivered_evidence_tokens'] == 0
    assert observer.remaining_evidence() == 10
    # A listing counts as a tool call but not as a query or read.
    second = observer.request_tools(0, [SimpleNamespace(function=SimpleNamespace(name='search', arguments={'query': 'x'}))])[0]
    assert observer.permit_tool(second)

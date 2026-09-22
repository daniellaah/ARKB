"""The links tool: navigate the vault's link graph without producing evidence."""
from types import SimpleNamespace

import pytest

from arkb.agent import AgentTools
from arkb.agent.observation import AgentBudget, AgentObserver
from arkb.agent.session import ToolSession
from arkb.knowledge.documents import DocumentAccess, load_notes
from arkb.knowledge.links import LinkGraph, resolve_links
from arkb.knowledge.models import EmbeddingSpec, IndexManifest, fingerprint_config
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.retrieval import ExactRetriever


@pytest.fixture
def linked_tools(tmp_path, linked_vault, engine):
    """Agent tools over the small vault, with its link graph in a snapshot."""
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1')
    with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        storage.create_build(
            IndexManifest(index_version='v1', vault_id='v', embedding_spec=spec,
                          chunking_fingerprint=fingerprint_config({}),
                          document_count=0, chunk_count=0),
            corpus_fingerprint='corpus', backend={'kind': 'qdrant'})
        storage.add_links('v1', resolve_links(load_notes(linked_vault)))
        documents = DocumentAccess(linked_vault, vault_id='v')
        with ExactRetriever(documents) as exact:
            yield AgentTools(documents=documents, exact=exact, engine=engine,
                             links=LinkGraph(storage, 'v1'))


def test_links_returns_both_directions_with_current_titles_and_context(linked_tools):
    result = linked_tools.links('Concepts/KV Cache.md')

    assert (result['source'], result['title']) == ('Concepts/KV Cache.md', 'KV Cache')
    assert [(link['source'], link['title']) for link in result['out']] == [
        ('Concepts/Attention.md', 'Attention'), ('Journal/2026-01-02.md', '2026-01-02')]
    assert [link['source'] for link in result['in']] == ['Journal/2026-01-02.md', 'index.md']
    # Enough context to see why the link is there, and how often it recurs.
    assert '[[Attention#Masking]]' in result['out'][0]['context']
    assert result['out'][0]['occurrences'] == 2 and 'occurrences' not in result['out'][1]
    assert (result['out_total'], result['in_total'], result['truncated']) == (2, 2, False)


def test_direction_and_limit_narrow_the_listing(linked_tools):
    assert set(linked_tools.links('index.md', direction='out')) == {
        'source', 'title', 'out', 'out_total', 'truncated'}
    incoming = linked_tools.links('index.md', direction='in')
    assert [link['source'] for link in incoming['in']] == ['Archive/KV Cache.md']

    bounded = linked_tools.links('Concepts/KV Cache.md', limit=1)
    assert len(bounded['out']) == len(bounded['in']) == 1
    assert (bounded['out_total'], bounded['in_total'], bounded['truncated']) == (2, 2, True)
    # A note nothing links to reports empty listings rather than an error.
    assert linked_tools.links('Concepts/Attention.md', direction='out')['out'] == []


def test_links_validates_its_arguments_and_needs_an_indexed_graph(linked_tools, documents, engine):
    for invalid in ({'direction': 'sideways'}, {'limit': 0}, {'limit': True}):
        with pytest.raises(ValueError):
            linked_tools.links('index.md', **invalid)
    with pytest.raises(LookupError):
        linked_tools.links('Concepts/Missing.md')

    without = AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine)
    assert 'links' not in {definition['name'] for definition in without.tool_definitions()}
    assert 'links' in {definition['name'] for definition in linked_tools.tool_definitions()}
    with pytest.raises(ValueError, match='no indexed link graph'):
        without.links('a.md')


def test_session_treats_links_as_navigation_and_never_as_evidence(linked_tools):
    session = ToolSession(linked_tools)
    assert 'links' in session._schemas and session.retrieval_attempted is False

    result = session.invoke('links', {'source': 'index.md', 'direction': 'out'})
    assert result['status'] == 'success' and [link['source'] for link in result['out']] == [
        'Concepts/KV Cache.md', 'Journal/2026-01-02.md']
    assert session.retrieval_attempted is True and session.references == {}
    assert session.invoke('links', {'source': 'gone.md'})['error']['code'] == 'source_unavailable'
    assert session.invoke('links', {'source': '../outside.md'})['error']['code'] == 'invalid_arguments'
    assert session.invoke('links', {'source': 'index.md', 'direction': 'up'})['error']['code'] == 'invalid_arguments'
    assert session.invoke('links', {})['error']['code'] == 'invalid_arguments'
    # Following links still leaves the answer needing cited evidence.
    error = session.invoke('finish', {'answer': 'the map links to the cache', 'status': 'answered',
                                      'evidence_refs': []})['error']
    assert error['code'] == 'invalid_citation'


def test_observer_charges_a_link_listing_as_one_tool_call_and_no_evidence(linked_tools):
    observer = AgentObserver(budget=AgentBudget(max_tool_calls=2, max_query_calls=1, max_read_calls=1,
                                                max_evidence_tokens=10, max_elapsed_ms=None),
                             counter=lambda text: len(text.split()), counter_identity='words')
    observer.start()
    session = ToolSession(linked_tools)
    call = SimpleNamespace(function=SimpleNamespace(name='links', arguments={'source': 'index.md'}))
    event = observer.request_tools(0, [call])[0]
    assert observer.permit_tool(event)
    observer.start_tool(event)

    assert observer.end_tool(event, session.invoke('links', call.function.arguments),
                             references=session.references)
    assert event['delivered_to_conversation'] and event['delivered_evidence_tokens'] == 0
    assert observer.remaining_evidence() == 10
    # It counts as a tool call, but leaves the query and read allowances alone.
    query = observer.request_tools(0, [SimpleNamespace(function=SimpleNamespace(
        name='search', arguments={'query': 'x'}))])[0]
    assert observer.permit_tool(query)

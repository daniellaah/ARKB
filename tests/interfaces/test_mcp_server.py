"""The MCP boundary, driven through the real protocol by an in-process client.

No Qdrant, Ollama or subprocess server: the transport is a memory stream pair,
the retrieval engine is scripted, and match runs against live files.
"""
import json
import re
from unittest.mock import Mock

import pytest

pytest.importorskip('mcp')

import anyio  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from arkb.agent import AgentTools  # noqa: E402
from arkb.interfaces.mcp_server import EVIDENCE_FIELDS, ReadOnlyTools, build_server  # noqa: E402
from arkb.knowledge.documents import DocumentAccess  # noqa: E402
from arkb.retrieval import ExactRetriever, RetrievalEngine, SearchResponse  # noqa: E402
from arkb.retrieval.models import chunk_result  # noqa: E402


BODY = 'Alpha body with foo() in it.\n\n## Detail\nrare idea about retrieval\n\n## Other\ntail text\n'


@pytest.fixture
def documents(tmp_path):
    nested = tmp_path / '04-Areas' / 'Career Development'
    nested.mkdir(parents=True)
    (nested / 'note.md').write_text(f'# Alpha\n\n{BODY}', encoding='utf-8')
    (tmp_path / 'flat.md').write_text('# Beta\n\nfoo() elsewhere\n', encoding='utf-8')
    return DocumentAccess(tmp_path, vault_id='v')


@pytest.fixture
def engine():
    engine = Mock(spec=RetrievalEngine)
    engine.candidate_k = 20
    engine.search.return_value = SearchResponse(query='question', method='semantic')
    return engine


@pytest.fixture
def tools(documents, engine):
    return AgentTools(documents=documents, exact=ExactRetriever(documents), engine=engine)


@pytest.fixture
def server(tools):
    return build_server(tools, vault_id='v')


def session(server, work):
    """Complete a handshake, run work against the client, and close the session."""
    async def connected():
        async with create_connected_server_and_client_session(server) as client:
            return await work(client)
    return anyio.run(connected)


def text(result):
    return '\n'.join(block.text for block in result.content)


def test_tool_list_is_read_only_and_omits_the_run_scoped_contract(server):
    listing = session(server, lambda client: client.list_tools())
    tools = {tool.name: tool for tool in listing.tools}
    assert set(tools) == {'list', 'match', 'search', 'read'}
    assert all(tool.annotations.readOnlyHint and not tool.annotations.destructiveHint
               and not tool.annotations.openWorldHint for tool in tools.values())
    read = tools['read'].inputSchema
    assert set(read['properties']) == {'source', 'expand', 'start_char'}
    assert read['required'] == ['source'] and read['properties']['expand']['enum'] == ['section', 'document']
    # Evidence references belong to one agent run; nothing advertised may promise them.
    advertised = ' '.join(f'{tool.description} {tool.inputSchema}' for tool in tools.values())
    assert not re.search(r'\bref\b|\brefs\b|ev_', advertised)
    # Search still advertises the modes the composed engine actually supports.
    assert tools['search'].inputSchema['properties']['mode']['enum'] == ['bm25', 'semantic', 'hybrid', None]


def test_search_returns_stateless_evidence(server, documents, engine):
    record = next(documents.records(source='04-Areas/Career Development/note.md'))
    hit = chunk_result(record, method='semantic', score=0.4, score_type='cosine')
    engine.search.return_value = SearchResponse(query='retrieval', method='semantic',
                                                results=(hit,), index_id='idx-1')
    result = session(server, lambda client: client.call_tool('search', {'query': 'retrieval', 'limit': 3}))
    engine.search.assert_called_once_with('retrieval', mode='hybrid', rerank=False, filters={}, top_k=3)
    assert result.isError is False
    assert result.structuredContent['query'] == 'retrieval'
    assert result.structuredContent['index_version'] == 'idx-1'
    found, = result.structuredContent['results']
    assert set(found) == set(EVIDENCE_FIELDS)
    assert found['source'] == '04-Areas/Career Development/note.md' and found['title'] == 'Alpha'
    assert found['document_revision'] == record.document_revision
    # The text block carries the same object, for clients that ignore structured content.
    assert json.loads(text(result)) == result.structuredContent


def test_match_finds_literal_text_in_live_files(server):
    result = session(server, lambda client: client.call_tool('match', {'query': 'foo()', 'limit': 5}))
    sources = [hit['source'] for hit in result.structuredContent['results']]
    assert sources == ['04-Areas/Career Development/note.md', 'flat.md']
    assert result.structuredContent['truncated'] is False
    assert all(set(hit) == set(EVIDENCE_FIELDS) for hit in result.structuredContent['results'])


def test_list_browses_notes_without_evidence(server):
    result = session(server, lambda client: client.call_tool('list', {'pattern': '*.md'}))
    assert [note['source'] for note in result.structuredContent['notes']] == [
        '04-Areas/Career Development/note.md', 'flat.md']
    assert result.structuredContent['truncated'] is False


def test_read_returns_a_whole_note_or_the_section_holding_an_offset(server):
    source = '04-Areas/Career Development/note.md'
    whole = session(server, lambda client: client.call_tool('read', {'source': source}))
    assert whole.structuredContent['result']['content'] == BODY.strip()
    assert whole.structuredContent['result']['start_char'] == 0
    anchor = BODY.index('rare idea')
    section = session(server, lambda client: client.call_tool(
        'read', {'source': source, 'expand': 'section', 'start_char': anchor}))
    result = section.structuredContent['result']
    assert result['content'] == '## Detail\nrare idea about retrieval\n\n'
    assert result['start_char'] == BODY.index('## Detail')
    assert set(result) == set(EVIDENCE_FIELDS)


@pytest.mark.parametrize('name, arguments, code', [
    ('search', {'query': 'x', 'limit': 0}, 'Input validation error'),
    ('search', {'query': 'x', 'unknown': 1}, 'Input validation error'),
    ('read', {'expand': 'document'}, 'Input validation error'),
    ('read', {'source': '../outside.md'}, 'invalid_arguments'),
    ('read', {'source': 'absent.md'}, 'source_unavailable'),
    ('match', {'query': '[', 'regex': True}, 'invalid_pattern'),
])
def test_bad_calls_are_tool_errors_and_the_session_survives(server, name, arguments, code):
    async def work(client):
        failed = await client.call_tool(name, arguments)
        return failed, await client.call_tool('list', {})

    failed, recovered = session(server, work)
    assert failed.isError is True and code in text(failed)
    assert recovered.isError is False and recovered.structuredContent['total'] == 2


def test_unknown_tool_and_hybrid_overshoot_are_reported(tools):
    vault = ReadOnlyTools(tools)
    with pytest.raises(ValueError) as unknown:
        vault.invoke('finish', {'answer': 'a', 'status': 'answered', 'evidence_refs': []})
    assert unknown.value.code == 'unknown_tool'
    with pytest.raises(ValueError) as overshoot:
        vault.invoke('search', {'query': 'x', 'mode': 'hybrid', 'limit': 50})
    assert overshoot.value.code == 'invalid_arguments'


@pytest.fixture
def linked_server(linked_vault, engine, tmp_path):
    """The same boundary over a vault whose snapshot carries a link graph."""
    from arkb.knowledge.documents import load_notes
    from arkb.knowledge.links import LinkGraph, resolve_links
    from arkb.knowledge.models import EmbeddingSpec, IndexManifest, fingerprint_config
    from arkb.knowledge.sqlite import SQLiteStorage

    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1')
    with SQLiteStorage(tmp_path / 'index.sqlite') as storage:
        storage.create_build(
            IndexManifest(index_version='v1', vault_id='v', embedding_spec=spec,
                          chunking_fingerprint=fingerprint_config({}), document_count=0, chunk_count=0),
            corpus_fingerprint='corpus', backend={'kind': 'qdrant'})
        storage.add_links('v1', resolve_links(load_notes(linked_vault)))
        documents = DocumentAccess(linked_vault, vault_id='v')
        with ExactRetriever(documents) as exact:
            tools = AgentTools(documents=documents, exact=exact, engine=engine,
                               links=LinkGraph(storage, 'v1'))
            yield build_server(tools, vault_id='v')


def test_the_boundary_republishes_the_new_filters_and_the_link_graph(linked_server):
    listing = session(linked_server, lambda client: client.list_tools())
    tools = {tool.name: tool for tool in listing.tools}
    assert set(tools) == {'list', 'links', 'match', 'search', 'read'}
    assert set(tools['list'].inputSchema['properties']) == {
        'pattern', 'tag', 'modified_after', 'modified_before', 'limit'}

    tagged = session(linked_server, lambda client: client.call_tool('list', {'tag': 'ai'}))
    assert [note['source'] for note in tagged.structuredContent['notes']] == [
        'Concepts/Attention.md', 'Concepts/KV Cache.md']
    followed = session(linked_server, lambda client: client.call_tool(
        'links', {'source': 'Concepts/KV Cache.md', 'direction': 'in'}))
    assert [link['source'] for link in followed.structuredContent['in']] == [
        'Journal/2026-01-02.md', 'index.md']
    missing = session(linked_server, lambda client: client.call_tool('links', {'source': 'gone.md'}))
    assert missing.isError is True and 'source_unavailable' in text(missing)

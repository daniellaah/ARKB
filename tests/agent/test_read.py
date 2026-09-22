from unittest.mock import Mock

import pytest

from arkb.agent import AgentTools
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval import ExactRetriever


def test_read_full_empty_and_open_ranges(tools, documents, engine):
    record = next(documents.records(source='a.md'))
    hit = tools.read(record.document_id)['result']
    assert hit['content'] == record.chunk.content and hit['title'] == 'Alpha'
    assert hit['document_id'] == record.document_id
    assert hit['chunk_id'] is None and hit['section_id'] is None
    assert (hit['start_char'], hit['end_char']) == (0, len(hit['content']))
    assert tools.read(record.document_id, end_char=2)['result']['content'] == 'éø'
    assert tools.read(record.document_id, start_char=3)['result']['content'] == hit['content'][3:]
    assert tools.read(record.document_id, start_char=2, end_char=2)['result']['content'] == ''
    empty = next(documents.records(source='empty.md'))
    assert tools.read(empty.document_id)['result']['content'] == ''
    engine.search.assert_not_called()


@pytest.mark.parametrize('options', [
    {'start_char': -1}, {'end_char': True}, {'start_char': 1.2}, {'start_char': '1'},
    {'end_char': 1000}, {'start_char': 4, 'end_char': 2}, {'section_id': ''},
    {'section_id': 'a' * 64, 'start_char': 0}, {'section_id': 'a' * 64, 'end_char': 1},
])
def test_read_invalid_inputs(tools, documents, options):
    with pytest.raises(ValueError):
        tools.read(next(documents.records()).document_id, **options)


@pytest.mark.parametrize('document_id', ['', ' ', None, True, 'not-an-id', '../a.md'])
def test_read_invalid_identifiers(tools, document_id):
    with pytest.raises(ValueError):
        tools.read(document_id)


def test_read_missing_document_section_and_deletion(tools, documents):
    with pytest.raises(LookupError):
        tools.read('0' * 64)
    record = next(documents.records())
    with pytest.raises(LookupError):
        tools.read(record.document_id, section_id='0' * 64)
    (documents.directory / record.chunk.source).unlink()
    with pytest.raises(LookupError):
        tools.read(record.document_id)


def test_read_delegates_once_and_propagates_errors(documents, engine):
    record = next(documents.records())
    access = Mock(spec=DocumentAccess)
    access.read.return_value = documents.read(record.document_id)
    exact = Mock(spec=ExactRetriever)
    tools = AgentTools(documents=access, exact=exact, engine=engine)
    tools.read(record.document_id, start_char=1, end_char=3)
    access.read.assert_called_once_with(record.document_id, source=None, section_id=None, start_char=1, end_char=3)
    error = PermissionError('cannot read')
    access.read.side_effect = error
    with pytest.raises(PermissionError) as raised:
        tools.read(record.document_id)
    assert raised.value is error
    engine.search.assert_not_called()
    exact.search.assert_not_called()


def test_read_known_source_directly_without_discovery(documents, engine):
    exact = Mock(spec=ExactRetriever)
    tools = AgentTools(documents=documents, exact=exact, engine=engine)
    record = next(documents.records(source='a.md'))
    assert tools.read(source='a.md') == tools.read(record.document_id)
    assert tools.read(record.document_id, source='a.md') == tools.read(record.document_id)
    assert tools.read(source='a.md', end_char=2)['result']['content'] == 'éø'
    engine.search.assert_not_called()
    exact.search.assert_not_called()


@pytest.mark.parametrize('options', [
    {}, {'source': ''}, {'source': ' '}, {'source': 1}, {'source': True},
])
def test_read_requires_a_valid_document_selector(tools, options):
    with pytest.raises(ValueError):
        tools.read(**options)


def test_read_rejects_conflicting_document_selectors(tools, documents):
    record = next(documents.records(source='a.md'))
    with pytest.raises(LookupError):
        tools.read(record.document_id, source='b.md')


def test_session_reads_nested_sources_and_rejects_paths_outside_the_vault(tmp_path):
    from arkb.agent.session import ToolSession
    from arkb.retrieval import RetrievalEngine, SearchResponse

    (tmp_path / '04-Areas').mkdir()
    (tmp_path / '04-Areas' / 'note.md').write_text('# Career\n\nA plan.', encoding='utf-8')
    (tmp_path / 'note.md').write_text('# Root\n\nAnother plan.', encoding='utf-8')
    (tmp_path.parent / 'outside.md').write_text('private', encoding='utf-8')
    documents = DocumentAccess(tmp_path, vault_id='v')
    engine = Mock(spec=RetrievalEngine)
    engine.search.return_value = SearchResponse(query='q', method='semantic')
    with ExactRetriever(documents) as exact:
        session = ToolSession(AgentTools(documents=documents, exact=exact, engine=engine))
        nested = session.invoke('read', {'source': '04-Areas/note.md'})
        assert nested['status'] == 'success'
        assert nested['result']['source'] == '04-Areas/note.md'
        assert nested['result']['content'] == 'A plan.'
        assert session.invoke('read', {'source': 'note.md'})['result']['content'] == 'Another plan.'
        for source in ('../outside.md', '/etc/passwd', '04-Areas\\note.md', 'C:/vault/note.md',
                       './04-Areas/note.md', '04-Areas/../04-Areas/note.md'):
            error = session.invoke('read', {'source': source})['error']
            assert error['code'] == 'invalid_arguments', source
            assert 'vault-relative' in error['message']
        assert session.invoke('read', {'source': '04-Areas/missing.md'})['error']['code'] == 'source_unavailable'
        assert session.invoke('match', {'query': 'plan', 'source': '../outside.md'})['error']['code'] == 'invalid_arguments'


def test_nested_listing_and_matching_report_vault_relative_sources(tmp_path):
    from arkb.agent.session import ToolSession
    from arkb.retrieval import RetrievalEngine, SearchResponse

    for relative in ('04-Areas/Career Development/plan.md', '01-Journal/plan.md'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Plan\n\nrare token', encoding='utf-8')
    (tmp_path / '.obsidian').mkdir()
    (tmp_path / '.obsidian' / 'workspace.md').write_text('rare token', encoding='utf-8')
    documents = DocumentAccess(tmp_path, vault_id='v')
    engine = Mock(spec=RetrievalEngine)
    engine.search.return_value = SearchResponse(query='q', method='semantic')
    with ExactRetriever(documents) as exact:
        session = ToolSession(AgentTools(documents=documents, exact=exact, engine=engine))
        listing = session.invoke('list', {})
        assert [note['source'] for note in listing['notes']] == [
            '01-Journal/plan.md', '04-Areas/Career Development/plan.md']
        matched = session.invoke('match', {'query': 'rare token', 'limit': 10})
        session.deliver(matched)
        hits = matched['results']
        assert [hit['source'] for hit in hits] == [
            '01-Journal/plan.md', '04-Areas/Career Development/plan.md']
        # A ref returned for one note expands to that note, not to its same-named sibling.
        expanded = session.invoke('read', {'ref': hits[1]['ref'], 'expand': 'document'})
        assert expanded['result']['source'] == '04-Areas/Career Development/plan.md'

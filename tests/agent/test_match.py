import json
from unittest.mock import Mock

import pytest

from arkb.agent import AgentTools
from arkb.retrieval import ExactRetriever, SearchResponse


def test_match_literal_content_filter_limit_and_unicode(tools, engine):
    result = tools.match('foo()', source='a.md', limit=1)
    assert result['query'] == 'foo()'
    hit, = result['results']
    assert (hit['source'], hit['title'], hit['content']) == ('a.md', 'Alpha', 'foo()')
    assert (hit['start_char'], hit['end_char']) == (8, 13)
    assert hit['chunk_id'] is None and hit['section_id'] is None
    assert json.loads(json.dumps(result)) == result
    engine.search.assert_not_called()


def test_match_regex_case_and_filename(tools):
    assert len(tools.match(r'foo\(\)', regex=True)['results']) == 2
    assert len(tools.match('foo()', case_sensitive=False)['results']) == 3
    hit, = tools.match(r'^a\.md$', target='source', regex=True)['results']
    assert hit['source'] == 'a.md' and hit['content'].startswith('éø')
    assert hit['start_char'] is None and hit['end_char'] is None
    assert tools.match('Alpha')['results'] == []  # Title is separate from the body.


@pytest.mark.parametrize('options', [{}, {'target': 'source'}, {'source': 'missing.md'}])
def test_match_empty_results(tools, options):
    assert tools.match('absent', **options) == {'query': 'absent', 'results': [], 'truncated': False}


@pytest.mark.parametrize('query, options', [
    ('', {}), ('  ', {}), (None, {}), ('x', {'limit': 0}), ('x', {'limit': True}),
    ('x', {'limit': 1.5}), ('x', {'source': ''}), ('x', {'source': 123}),
    ('x', {'target': 'title'}), ('x', {'target': []}), ('x', {'regex': 1}),
    ('x', {'case_sensitive': 'yes'}), ('[', {'regex': True}), ('a\0b', {}),
])
def test_match_invalid_inputs(tools, query, options):
    with pytest.raises(ValueError):
        tools.match(query, **options)


def test_match_delegates_once_and_propagates_failure(documents, engine):
    exact = Mock(spec=ExactRetriever)
    exact.search.return_value = SearchResponse(query='term', method='exact')
    tools = AgentTools(documents=documents, exact=exact, engine=engine)
    assert tools.match('term', target='source', regex=True, case_sensitive=False,
                       source='b.md', limit=3)['results'] == []
    exact.search.assert_called_once_with('term', target='source', regex=True,
                                        case_sensitive=False, filters={'source': 'b.md'}, top_k=3, timeout=30,
                                        unique_sources=False)
    error = OSError('reader failed')
    exact.search.side_effect = error
    with pytest.raises(OSError) as raised:
        tools.match('term')
    assert raised.value is error


def test_match_to_read_and_live_edits(tools, documents):
    hit = tools.match('foo()', source='a.md')['results'][0]
    assert tools.read(hit['document_id'], start_char=hit['start_char'],
                      end_char=hit['end_char'])['result']['content'] == 'foo()'
    assert tools.read(hit['document_id'])['result']['content'].startswith('éø café')
    (documents.directory / 'a.md').write_text('# Changed\n\nnew text', encoding='utf-8')
    assert tools.match('foo()', source='a.md')['results'] == []
    current = tools.match('new text')['results'][0]
    assert current['document_id'] == hit['document_id']
    assert current['document_revision'] != hit['document_revision']
    assert tools.read(hit['document_id'])['result']['content'] == 'new text'


def test_match_unique_sources_and_truncation_reach_the_tool_observation(tools):
    from arkb.agent.session import ToolSession
    occurrences = tools.match('foo()', limit=2)
    assert [h['source'] for h in occurrences['results']] == ['a.md', 'b.md'] and occurrences['truncated'] is False
    assert tools.match('foo()', case_sensitive=False, limit=2)['truncated'] is True
    unique = tools.match('foo', unique_sources=True, limit=1)
    assert [h['source'] for h in unique['results']] == ['a.md'] and unique['truncated'] is True
    assert tools.match('foo', unique_sources=True, limit=2)['truncated'] is False
    session = ToolSession(tools)
    observed = session.invoke('match', {'query': 'foo', 'unique_sources': True, 'limit': 1})
    assert observed['status'] == 'success' and observed['truncated'] is True
    assert {h['source'] for h in observed['results']} == {'a.md'}
    assert 'truncated' not in session.invoke('search', {'query': 'x'})
    schema = next(d for d in session.definitions if d['name'] == 'match')
    assert schema['parameters']['properties']['unique_sources'] == {
        'type': 'boolean', 'default': False,
        'description': 'At most one result per note; use it to enumerate matching notes completely.'}
    assert 'truncated' in schema['description']
    assert session.invoke('match', {'query': 'foo', 'unique_sources': 'yes'})['status'] == 'recoverable_error'


def test_unique_sources_defaults_to_a_larger_limit_than_occurrences(documents, engine):
    exact = Mock(spec=ExactRetriever)
    exact.search.return_value = SearchResponse(query='term', method='exact', truncated=False)
    tools = AgentTools(documents=documents, exact=exact, engine=engine)
    tools.match('term')
    assert exact.search.call_args.kwargs['top_k'] == 5
    tools.match('term', unique_sources=True)
    assert exact.search.call_args.kwargs['top_k'] == 50
    tools.match('term', unique_sources=True, limit=7)
    assert exact.search.call_args.kwargs['top_k'] == 7

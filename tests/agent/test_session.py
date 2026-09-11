"""Public run-session contract, backed by real live files and ranked records."""
import pytest
from arkb.agent.session import ToolSession
from arkb.retrieval import SearchResponse
from arkb.retrieval.models import chunk_result


def test_reference_binds_search_evidence_and_resolves_live_document(tools, engine, documents):
    record = next(documents.records())
    engine.search.return_value = SearchResponse(query='rare', method='semantic', index_id='snapshot',
        results=(chunk_result(record, method='semantic', score=.5, score_type='cosine_similarity'),))
    session = ToolSession(tools)
    result = session.invoke('search', {'query': 'rare'})
    assert result['status'] == 'success'
    hit, = result['results']
    assert set(hit) == {'ref', 'source', 'title', 'content'}
    assert session.references[hit['ref']]['document_id'] == record.document_id
    session.deliver(result)
    read = session.invoke('read', {'ref': hit['ref']})
    assert read['status'] == 'success'
    assert read['result']['source'] == 'a.md'
    assert read['result']['content'] == documents.read(source='a.md').content
    assert session.resolutions[-1]['ref'] == hit['ref']
    assert session.invoke('read', {'ref': hit['ref'], 'source': 'b.md'})['status'] == 'recoverable_error'


@pytest.mark.parametrize('change', ['edit', 'delete', 'rename'])
def test_stale_deleted_renamed_references_never_leak(tools, documents, change):
    session = ToolSession(tools)
    result = session.invoke('match', {'query': 'café'})
    session.deliver(result)
    ref = result['results'][0]['ref']
    path = documents.directory / 'a.md'
    if change == 'edit': path.write_text('# Different\nother content')
    elif change == 'delete': path.unlink()
    else: path.rename(documents.directory / 'renamed.md')
    outcome = session.invoke('read', {'ref': ref})
    assert outcome['status'] == 'recoverable_error'
    assert outcome['error']['code'] == ('stale_reference' if change == 'edit' else 'source_unavailable')
    assert 'content' not in outcome
    assert session.invoke('finish', {'answer': 'old text', 'status': 'answered',
                                   'evidence_refs': [ref]})['status'] == 'recoverable_error'
    if change == 'edit':
        fresh = session.invoke('read', {'source': 'a.md'})
        assert fresh['result']['content'] == 'other content'
        assert fresh['result']['ref'] != ref


@pytest.mark.parametrize('name,args,code', [
    ('read', {'ref': 'ev_unknown'}, 'invalid_reference'),
    ('read', {'source': 'a.md', 'expand': 'invalid'}, 'invalid_arguments'),
    ('read', {'source': '../a.md'}, 'invalid_arguments'),
    ('read', {'document_id': 'a.md'}, 'invalid_arguments'),
    ('read', {'source': 'a.md', 'start_char': 999}, 'invalid_arguments'),
    ('read', {}, 'invalid_arguments'),
    ('match', {'query': '[', 'regex': True}, 'invalid_pattern'),
    ('search', {'query': 'x', 'limit': True}, 'invalid_arguments'),
    ('search', {'query': 'x', 'mode': 'bogus'}, 'invalid_arguments'),
    ('search', {'query': 'x', 'limit': -2}, 'invalid_arguments'),
    ('search', {'query': ' '}, 'invalid_arguments'),
    ('search', 'not an object', 'invalid_arguments'),
    ('unknown', {}, 'unknown_tool'),
])
def test_expected_misuse_is_structured(tools, name, args, code):
    result = ToolSession(tools).invoke(name, args)
    assert result['status'] == 'recoverable_error'
    assert result['error']['code'] == code


def test_reference_run_scope_withheld_evidence_and_stable_repeats(tools):
    session = ToolSession(tools)
    first = session.invoke('match', {'query': 'café'})
    ref = first['results'][0]['ref']
    assert session.invoke('read', {'ref': ref})['error']['code'] == 'undelivered_reference'
    session.deliver(first)
    assert session.invoke('match', {'query': 'café'}) == first
    assert ToolSession(tools).invoke('read', {'ref': ref})['error']['code'] == 'invalid_reference'
    snippet = session.invoke('read', {'ref': ref, 'expand': 'snippet'})
    assert snippet['result']['content'] == 'café'
    internal = session.references[snippet['result']['ref']]
    assert internal['end_char'] - internal['start_char'] == len('café')


def test_backend_invariant_is_not_classified_as_agent_misuse(tools, engine):
    engine.search.side_effect = ValueError('corrupt snapshot')
    with pytest.raises(ValueError, match='corrupt snapshot'):
        ToolSession(tools).invoke('search', {'query': 'x'})


def test_internal_lookup_bug_is_fatal_not_source_unavailable(tools, monkeypatch):
    def broken(**arguments):
        raise KeyError('internal mapping')
    monkeypatch.setattr(tools, 'read', broken)
    with pytest.raises(KeyError):
        ToolSession(tools).invoke('read', {'source': 'a.md'})


def test_empty_retrieval_requires_insufficient_evidence(tools):
    session = ToolSession(tools)
    assert session.invoke('match', {'query': 'not present'})['results'] == []
    args = {'answer': 'unsupported fact', 'status': 'answered', 'evidence_refs': []}
    assert session.invoke('finish', args)['error']['code'] == 'invalid_citation'
    assert session.invoke('finish', {**args, 'status': 'insufficient_evidence'})['status'] == 'success'


def test_wrong_vault_search_result_cannot_resolve_even_with_existing_source(tools, engine, documents):
    from arkb.knowledge.documents import DocumentAccess
    foreign = next(DocumentAccess(documents.directory, vault_id='other-vault').records())
    engine.search.return_value = SearchResponse(query='x', method='semantic',
        results=(chunk_result(foreign, method='semantic', score=.5, score_type='cosine_similarity'),))
    session = ToolSession(tools)
    found = session.invoke('search', {'query': 'x'}); session.deliver(found)
    outcome = session.invoke('read', {'ref': found['results'][0]['ref']})
    assert outcome['status'] == 'recoverable_error' and outcome['error']['code'] == 'source_unavailable'


def test_search_depth_misuse_is_recoverable_before_backend(tools, engine):
    engine.candidate_k = 20
    result = ToolSession(tools).invoke('search', {'query': 'x', 'mode': 'hybrid', 'limit': 21})
    assert result['status'] == 'recoverable_error' and result['error']['code'] == 'invalid_arguments'
    engine.search.assert_not_called()

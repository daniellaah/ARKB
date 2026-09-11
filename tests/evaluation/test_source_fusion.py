"""Phase C source ranking contract, independent of labels and retrieval providers."""
from dataclasses import replace
import pytest
from arkb.retrieval.models import SearchResult, SearchResponse
from arkb.evaluation.source_fusion import fuse_sources


def hit(source, chunk, *, content='verbatim', index='snapshot', revision='revision'):
    return SearchResult(source_id=source, source=source+'.md', chunk_id=chunk,
        content=content, start_char=7, end_char=7+len(content), method='leg', score=2., score_type='raw',
        metadata={'index_version':index,'document_revision':revision,'vault_id':'vault',
                  'title':'Title','section_id':'section'})


def legs(bm25=(), semantic=()):
    return {name:SearchResponse(query='question',method=name,results=tuple(hits),index_id='snapshot')
            for name,hits in (('bm25',bm25),('semantic',semantic))}


def test_c1_compacts_source_ranks_and_uses_only_best_leg_chunk():
    a1,a2,b=hit('a','a1'),hit('a','a2'),hit('b','b')
    result=fuse_sources(legs([a1,a2,b],[a2]),policy='C1',top_k=10)
    assert [h.source_id for h in result.results]==['a','b']
    assert [h.score for h in result.results]==pytest.approx([.03278688524590164,.016129032258064516])
    assert result.results[0].chunk_id=='a1'
    votes=result.results[1].metadata['source_fusion']['legs']
    assert votes[0]['source_rank']==2
    assert votes[0]['chunks'][0]['chunk_rank']==3
    assert result.index_id=='snapshot'


def test_c2_uses_two_raw_chunk_ranks_then_compact_source_rrf():
    a,b1,b2,b3=hit('a','a'),hit('b','b1'),hit('b','b2'),hit('b','b3')
    response=fuse_sources(legs([a,b1,b2,b3]),policy='C2',top_k=100)
    assert [h.source_id for h in response.results]==['b','a']
    assert response.results[0].score==pytest.approx(.01639344262295082)
    vote=response.results[0].metadata['source_fusion']['legs'][0]
    assert [c['chunk_rank'] for c in vote['chunks']]==[2,3]
    assert vote['aggregation_score']==pytest.approx(.03200204813108039)
    assert response.results[0].chunk_id=='b1'
    without_extra=fuse_sources(legs([a,b1,b2]),policy='C2',top_k=100)
    assert response==without_extra


@pytest.mark.parametrize('policy',['C1','C2'])
def test_shared_chunk_accepts_leg_specific_metadata_and_retains_verbatim_evidence(policy):
    h=hit('source','chunk',content='An exact\n  excerpt.')
    lexical=replace(h,method='bm25',metadata={**h.metadata,'bm25':{'k1':1.2}})
    semantic=replace(h,method='semantic',score=.3,score_type='cosine_similarity')
    result=fuse_sources(legs([lexical],[semantic]),policy=policy).results[0]
    assert (result.source_id,result.chunk_id,result.content,result.start_char,result.end_char)==(
        h.source_id,h.chunk_id,'An exact\n  excerpt.',7,26)
    assert all(result.metadata[key]==h.metadata[key] for key in h.metadata)
    votes=result.metadata['source_fusion']['legs']
    assert [v['chunks'][0]['raw_score'] for v in votes]==[2.,.3]
    assert sum(c['contribution'] for v in votes for c in v['chunks'])==pytest.approx(result.score)


@pytest.mark.parametrize('policy',['C1','C2'])
@pytest.mark.parametrize('pattern',[[],['z'],['z','z','z'],['z','a','b'],['z','a','z','b','a']])
def test_source_subset_uniqueness_and_repeated_replay(policy,pattern):
    hits=[hit(source,str(i)) for i,source in enumerate(pattern)]
    response=fuse_sources(legs(hits,list(reversed(hits))),policy=policy)
    assert {h.source_id for h in response.results}==set(pattern)
    assert len(response.results)==len(set(pattern))
    assert response==fuse_sources(legs(hits,list(reversed(hits))),policy=policy)
    for result in response.results:
        assert result.identity in {h.identity for h in hits}
        assert result.metadata['source_fusion']['rrf_k']==60
        assert result.score==pytest.approx(sum(v['vote'] for v in result.metadata['source_fusion']['legs']))


@pytest.mark.parametrize('policy',['C1','C2'])
def test_equal_scores_preserve_incoming_source_order_instead_of_identity(policy):
    z,a=hit('z','z'),hit('a','a')
    original=legs([z,a],[a,z])
    assert [h.source_id for h in fuse_sources(original,policy=policy).results]==['z','a']
    assert fuse_sources(original,policy=policy)==fuse_sources(dict(reversed(list(original.items()))),policy=policy)


@pytest.mark.parametrize('policy',['C1','C2'])
def test_filters_and_source_top_k_are_applied_correctly(policy):
    z,a=hit('z','z'),hit('a','a')
    inputs=legs([z,z,a],[z,a])
    assert [h.source_id for h in fuse_sources(inputs,policy=policy,top_k=1).results]==['z']
    assert [h.source_id for h in fuse_sources(inputs,policy=policy,top_k=1,filters={'source':'a.md'}).results]==['a']
    assert fuse_sources(inputs,policy=policy,filters={'source':'missing.md'}).results==()
    with pytest.raises(ValueError):fuse_sources(inputs,policy=policy,top_k=0)
    with pytest.raises(ValueError):fuse_sources(inputs,policy=policy,filters={'unknown':'a'})


@pytest.mark.parametrize('policy',['C1','C2'])
def test_duplicate_identity_votes_once_and_keeps_original_positions(policy):
    a,b=hit('a','a'),hit('b','b')
    result=fuse_sources(legs([a,a,b]),policy=policy)
    assert len(result.results[0].metadata['source_fusion']['legs'][0]['chunks'])==1
    assert result.results[1].metadata['source_fusion']['legs'][0]['chunks'][0]['chunk_rank']==3


@pytest.mark.parametrize('policy',['C1','C2'])
@pytest.mark.parametrize('change',['snapshot','revision','content','source','coordinates','section'])
def test_inconsistent_captured_evidence_is_rejected(policy,change):
    a=hit('a','chunk')
    if change=='snapshot':other=hit('a','other',index='different')
    elif change=='revision':other=hit('a','other',revision='different')
    elif change=='content':other=hit('a','chunk',content='modified')
    elif change=='source':other=replace(a,source='different.md')
    elif change=='coordinates':other=replace(a,start_char=8,end_char=16)
    else:other=replace(a,metadata={**a.metadata,'section_id':'different'})
    with pytest.raises(ValueError):fuse_sources(legs([a],[other]),policy=policy)


@pytest.mark.parametrize('change',['query','index','missing_index'])
def test_leg_responses_must_share_query_and_snapshot(change):
    responses=legs([hit('a','a')])
    kwargs={'query':'different'} if change=='query' else {'index_id':None if change=='missing_index' else 'different'}
    responses['semantic']=replace(responses['semantic'],**kwargs)
    with pytest.raises(ValueError):fuse_sources(responses,policy='C1')


def test_invalid_policy_or_unnamed_legs_are_rejected():
    with pytest.raises(ValueError):fuse_sources(legs(),policy='learned')
    with pytest.raises(ValueError):fuse_sources({},policy='C1')
    with pytest.raises(ValueError):fuse_sources({'':legs()['bm25']},policy='C1')

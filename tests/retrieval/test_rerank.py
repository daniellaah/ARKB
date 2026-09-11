from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from arkb.retrieval import SearchResult


def candidates():
    return tuple(SearchResult(source_id=name, source=name + '.md', chunk_id=name,
                              content=text, method='semantic', score=score,
                              score_type='cosine_similarity', metadata={'title': name})
                 for name, text, score in [('a', 'Dogs bark.', .9), ('b', 'Paris is in France.', .4)])


def test_reranker_reorders_frozen_candidates_and_retains_input_rank_score_and_text():
    from arkb.retrieval.rerank import Reranker
    scorer = SimpleNamespace(identity='frozen-model-v1', score_type='relevance_logit',
                             score=Mock(return_value=[-2., 5.]))
    before = candidates()
    reranker = Reranker(scorer)
    result = reranker.rerank('Where is Paris?', before, top_k=1)
    assert result[0].chunk_id == 'b' and result[0].content == before[1].content
    assert result[0].score == 5. and result[0].score_type == 'relevance_logit'
    provenance = result[0].metadata['rerank']
    assert (provenance['input_rank'], provenance['input_score'], provenance['input_score_type']) == (2, .4, 'cosine_similarity')
    assert provenance['scorer'] == 'frozen-model-v1'
    assert 'rerank' not in before[1].metadata
    scorer.score.assert_called_once_with('Where is Paris?', before)
    assert reranker.rerank('Where is Paris?', before, top_k=1) == result


def test_reranker_empty_ties_invalid_scores_and_duplicate_identity():
    from arkb.retrieval.rerank import Reranker
    scorer = SimpleNamespace(identity='fixed', score_type='logit', score=Mock(return_value=[1., 1.]))
    reranker = Reranker(scorer)
    assert reranker.rerank('query', []) == ()
    scorer.score.assert_not_called()
    assert [h.chunk_id for h in reranker.rerank('query', list(reversed(candidates())))] == ['b', 'a']
    for scores in ([1.], [True, 1.], [float('nan'), 1.], [[1.], [2.]]):
        scorer.score.return_value = scores
        result = reranker.rerank('query', candidates())
        assert [h.identity for h in result] == [h.identity for h in candidates()]
        assert [h.score for h in result] == [.9, .4]
        assert all(h.metadata['rerank']['fallback'] == 'invalid_scores' for h in result)
    with pytest.raises(ValueError, match='duplicate'):
        reranker.rerank('query', [candidates()[0]] * 2)
    with pytest.raises(ValueError):
        reranker.rerank(' ', [])


def test_frozen_candidate_evaluation_isolates_ranking_changes_and_latency():
    from arkb.retrieval.rerank import Reranker
    from arkb.evaluation.retrieval import evaluate_reranker
    scorer = SimpleNamespace(identity='frozen', score_type='logit', score=Mock(return_value=[-2., 5.]))
    row = evaluate_reranker(Reranker(scorer), 'Paris?', candidates(), {'b.md': 1}, top_k=1)
    assert row['before']['recall_at_k'] == 0
    assert row['after']['recall_at_k'] == 1
    assert row['rank_changes'][0] == {'identity': ['b', 'chunk', 'b'], 'before': 2, 'after': 1}
    assert row['latency_ms'] >= 0
    assert len(row['candidates']) == 2 and len(row['results']) == 1


@pytest.mark.parametrize('scores, expected', [
    ([2., 1., 2., 1.], ['z', 'b', 'y', 'a']),
    ([3., 3., 3., 3.], ['z', 'y', 'b', 'a']),
    ([1., 2., 3., 4.], ['a', 'b', 'y', 'z']),
])
def test_score_groups_preserve_upstream_order_and_every_candidate(scores, expected):
    from dataclasses import replace
    from arkb.retrieval.rerank import Reranker
    before = tuple(replace(candidates()[0], source_id=name, source=name+'.md', chunk_id=name)
                   for name in ('z', 'y', 'b', 'a'))
    scorer = SimpleNamespace(identity='frozen', score_type='logit', score=lambda *_: scores)
    after = Reranker(scorer).rerank('query', before)
    assert [h.chunk_id for h in after] == expected
    assert {h.identity for h in after} == {h.identity for h in before}
    original = {h.identity: h for h in before}
    assert all(h.content == original[h.identity].content for h in after)


def test_distinct_chunks_of_one_source_remain_distinct_candidates():
    from dataclasses import replace
    from arkb.retrieval.rerank import Reranker
    first = candidates()[0]
    second = replace(first, chunk_id='second-chunk', content='Other evidence.')
    scorer = SimpleNamespace(identity='frozen', score_type='logit', score=lambda *_: [0., 1.])
    after = Reranker(scorer).rerank('query', [first, second])
    assert [h.chunk_id for h in after] == ['second-chunk', first.chunk_id]
    assert [h.source for h in after] == [first.source, first.source]


@pytest.mark.parametrize('error', [RuntimeError('model unavailable'), OSError('weights unreadable')])
def test_scoring_execution_failure_preserves_original_scores_and_ranking(error):
    from arkb.retrieval.rerank import Reranker
    scorer = SimpleNamespace(identity='frozen', score_type='logit', score=Mock(side_effect=error))
    before = tuple(reversed(candidates()))
    after = Reranker(scorer).rerank('query', before)
    assert [h.identity for h in after] == [h.identity for h in before]
    assert [(h.method, h.score, h.score_type) for h in after] == [(h.method, h.score, h.score_type) for h in before]
    assert all(h.metadata['rerank']['fallback'] == 'scoring_error' for h in after)
    assert type(error).__name__ in after[0].metadata['rerank']['message']


@pytest.mark.parametrize('error', [KeyError('broken invariant'), TypeError('programming error'), ValueError('invalid configuration')])
def test_programming_and_configuration_errors_are_not_hidden(error):
    from arkb.retrieval.rerank import Reranker
    scorer = SimpleNamespace(identity='frozen', score_type='logit', score=Mock(side_effect=error))
    with pytest.raises(type(error)):
        Reranker(scorer).rerank('query', candidates())


@pytest.mark.parametrize('scores', [None, 5., ['a', 'b'], [float('inf'), 1.]])
def test_unusable_score_containers_have_explicit_safe_fallback(scores):
    from arkb.retrieval.rerank import Reranker
    scorer = SimpleNamespace(identity='frozen', score_type='logit', score=lambda *_: scores)
    after = Reranker(scorer).rerank('query', candidates(), top_k=1)
    assert after[0].identity == candidates()[0].identity
    assert after[0].metadata['rerank']['fallback'] == 'invalid_scores'

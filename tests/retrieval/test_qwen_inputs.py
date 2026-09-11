"""Public token-allocation boundary with a real local byte-level tokenizer."""

from dataclasses import replace

import pytest

from tests.retrieval.test_rerank import candidates


@pytest.fixture
def tokenizer():
    transformers = pytest.importorskip('transformers')
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    from arkb.retrieval.qwen_rerank import PREFIX, SUFFIX, INSTRUCTION
    backend.train_from_iterator(['A short English query. A technical document.',
                                 '\n<Document>: \n\nDogs bark.', PREFIX, SUFFIX, INSTRUCTION],
        trainers.BpeTrainer(vocab_size=1000, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]')


def test_current_policy_reconstructs_official_tokens_and_measures_body_loss(tokenizer):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    from arkb.retrieval.qwen_rerank import INSTRUCTION, PREFIX, SUFFIX
    hit = candidates()[0]
    query = 'query ' * 1000
    result = QwenInputBuilder(tokenizer, max_length=512).prepare(query, hit)
    pair = f'<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: a\n\nDogs bark.'
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    legacy = tokenizer(pair, truncation='longest_first', max_length=512-len(prefix)-len(suffix))['input_ids']
    assert result['input_ids'] == prefix + legacy + suffix
    assert result['body_tokens_before'] > 0 and result['body_tokens_retained'] == 0
    assert result['truncated'] and result['body_empty']
    assert result['total_model_tokens'] == 512


def test_empty_title_does_not_count_document_separator_as_title(tokenizer):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    hit = replace(candidates()[0], metadata={'title': ''})
    row = QwenInputBuilder(tokenizer).prepare('English question?', hit)
    assert row['title_tokens_before'] == row['title_tokens_retained'] == 0


@pytest.mark.parametrize('query_cap', [128, 192])
def test_allocations_protect_body_and_both_ends_of_a_long_query(tokenizer, query_cap):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    hit = replace(candidates()[0], content='Document evidence. ' * 500,
                  metadata={'title': 'Long technical document title. ' * 100})
    query = 'PROBLEM_START ' + 'example ' * 800 + ' FINAL_QUESTION'
    row = QwenInputBuilder(tokenizer, query_cap=query_cap, title_cap=64).prepare(query, hit)
    assert row['query_tokens_retained'] + row['query_gap_tokens'] <= query_cap
    assert row['title_tokens_retained'] <= 64
    assert row['body_tokens_retained'] >= 128
    assert row['total_model_tokens'] <= 512
    visible = tokenizer.decode(row['input_ids'])
    assert 'PROBLEM_START' in visible and 'FINAL_QUESTION' in visible
    assert '\n\nDocument evidence.' in visible
    assert row['truncated'] and not row['body_empty']


def test_short_input_is_byte_for_byte_unchanged(tokenizer):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    legacy = QwenInputBuilder(tokenizer).prepare('English question?', candidates()[0])
    allocated = QwenInputBuilder(tokenizer, query_cap=128, title_cap=64).prepare('English question?', candidates()[0])
    assert allocated['input_ids'] == legacy['input_ids']
    assert allocated['truncated'] is False
    assert allocated['body_retained_fraction'] == 1


@pytest.mark.parametrize('cap', [128, 129, 191, 192])
def test_unicode_byte_boundaries_and_technical_delimiters_remain_valid(tokenizer, cap):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    query = 'START ' + '🙂 café e\u0301 x->y() <tag> ' * 200 + ' END'
    hit = replace(candidates()[0], content='🙂 evidence e\u0301 λ(x) ' * 300,
                  metadata={'title': '🙂 Title ' * 100})
    row = QwenInputBuilder(tokenizer, query_cap=cap, title_cap=64).prepare(query, hit)
    decoded = tokenizer.decode(row['input_ids'])
    assert '\ufffd' not in decoded
    assert '\n<Document>: ' in decoded and 'START' in decoded and ' END' in decoded
    assert row['total_model_tokens'] <= 512
    assert row['query_tokens_retained'] + row['query_gap_tokens'] <= cap
    assert row['body_tokens_retained'] >= 128


def test_impossible_body_reservation_is_explicit(tokenizer):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    builder = QwenInputBuilder(tokenizer, max_length=200, query_cap=128, title_cap=64)
    with pytest.raises(ValueError, match='minimum body'):
        builder.prepare('A long query. '*200, replace(candidates()[0], content='Evidence. '*500))


@pytest.mark.parametrize('kwargs', [
    {'query_cap': 128}, {'title_cap': 64}, {'query_cap': 0, 'title_cap': 64},
    {'query_cap': True, 'title_cap': 64}, {'query_strategy': 'unknown'},
])
def test_invalid_allocation_configuration_is_rejected(tokenizer, kwargs):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    with pytest.raises(ValueError):
        QwenInputBuilder(tokenizer, **kwargs)


@pytest.mark.parametrize('empty_first_batch', [False, True])
def test_scorer_sends_exact_instrumented_tokens_to_model(tokenizer, monkeypatch, empty_first_batch):
    from types import SimpleNamespace
    from arkb.retrieval.qwen_rerank import QwenRerankerScorer, PREFIX, SUFFIX
    torch = pytest.importorskip('torch')
    import transformers
    tokenizer.add_tokens(['yes', 'no'])
    tokenizer.padding_side = 'left'
    received = []; batches = []
    class Model:
        def to(self, *_): return self
        def eval(self): return self
        def __call__(self, input_ids, attention_mask, **kwargs):
            batches.append(len(input_ids))
            received.extend(ids[mask.bool()].tolist() for ids, mask in zip(input_ids, attention_mask))
            assert attention_mask[:, -1].tolist() == [1] * len(input_ids)
            scores = torch.zeros((len(input_ids), 1, len(tokenizer)))
            scores[:, 0, tokenizer.convert_tokens_to_ids('yes')] = 3.
            return SimpleNamespace(logits=scores)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: Model())
    scorer = QwenRerankerScorer(query_cap=128, title_cap=64, batch_size=2)
    hits = (replace(candidates()[0], content='Evidence '*500), candidates()[1],
            replace(candidates()[0], source_id='third', source='third.md', chunk_id='third', content='Short evidence.'))
    if empty_first_batch:
        hits = tuple(replace(h, content='') for h in hits[:2]) + hits[2:]
    query = 'Beginning question. ' + 'example '*500 + ' Final question?'
    details = scorer.prepare_inputs(query, hits)
    assert scorer.score(query, hits) == [3., 3., 3.]
    assert batches == [2, 1]
    assert received == [r['input_ids'] for r in details]
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    for ids in received:
        assert len(ids) <= 512
        assert ids[:len(prefix)] == prefix
        assert ids[-len(suffix):] == suffix


def test_source_identity_and_unrelated_metadata_cannot_change_model_input(tokenizer):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    builder = QwenInputBuilder(tokenizer, query_cap=128, title_cap=64)
    first = candidates()[0]
    other = replace(first, source_id='different-document', source='different.md', chunk_id='different-chunk',
                    score=.01, metadata={**first.metadata, 'unrelated_annotation': 'must not enter model input'})
    assert builder.prepare('A question?', first)['input_ids'] == builder.prepare('A question?', other)['input_ids']


@pytest.mark.parametrize('legacy_truncation', [False, True])
def test_all_body_empty_preserves_hybrid_without_calling_model(tokenizer, monkeypatch, legacy_truncation):
    from unittest.mock import Mock
    pytest.importorskip('torch')
    import transformers
    from arkb.retrieval.qwen_rerank import QwenRerankerScorer
    from arkb.retrieval.rerank import Reranker
    tokenizer.add_tokens(['yes', 'no'])
    tokenizer.padding_side = 'left'
    model = Mock(side_effect=AssertionError('Empty input must not invoke model inference.'))
    model.to.return_value = model.eval.return_value = model
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **k: tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: model)
    scorer = QwenRerankerScorer(query_cap=None if legacy_truncation else 128)
    query = 'query ' * 1000 if legacy_truncation else 'An English question?'
    hits = tuple(reversed(candidates())) if legacy_truncation else tuple(
        replace(h, content='') for h in reversed(candidates()))
    assert all(d['body_empty'] for d in scorer.prepare_inputs(query, hits))
    after = Reranker(scorer).rerank(query, hits)
    model.assert_not_called()
    assert [(h.identity, h.content, h.method, h.score, h.score_type) for h in after] == [
        (h.identity, h.content, h.method, h.score, h.score_type) for h in hits]
    assert all(h.metadata['rerank']['fallback'] == 'empty_body' for h in after)


@pytest.mark.parametrize('query_cap', [1, 3])
def test_query_budget_cannot_silently_remove_every_complete_character(tokenizer, query_cap):
    from arkb.retrieval.qwen_inputs import QwenInputBuilder
    builder = QwenInputBuilder(tokenizer, query_cap=query_cap, title_cap=64)
    with pytest.raises(ValueError, match='query allocation retains no query'):
        builder.prepare('🙂' * 100, candidates()[0])

"""Blinded review sheet: hash-chosen sample, per-question shuffled arms, mapping kept apart."""
from evaluation.devloop.review_sheet import hashed, markdown, sample


def test_sample_is_by_id_hash_only_and_order_independent():
    scenarios = [{'id': f'q{i}', 'slice': 'v2' if i % 2 else 'musique'} for i in range(40)]
    chosen = sample(scenarios, 'v2', 5, 'salt')
    assert len(chosen) == 5 and all(s['slice'] == 'v2' for s in chosen)
    assert [s['id'] for s in chosen] == [s['id'] for s in sample(list(reversed(scenarios)), 'v2', 5, 'salt')]
    assert [s['id'] for s in chosen] != [s['id'] for s in sample(scenarios, 'v2', 5, 'other-salt')]
    assert hashed('a', 'b') != hashed('a', 'c')


def test_markdown_hides_arms_and_quotes_answers():
    items = [{'id': 'q1', 'slice': 'musique', 'query': 'Who?', 'query_zh': None,
              'reference': {'answerable': True, 'gold_answer': 'X', 'answer_aliases': [], 'support_idxs': [1]},
              'answers': [{'letter': 'A', 'status': 'answered', 'answer': 'X\nbecause', 'cited_sources': ['a.md'], 'error': None},
                          {'letter': 'B', 'status': 'error', 'answer': None, 'cited_sources': [], 'error': {'type': 'E'}}],
              'review': {}}]
    text = markdown(items)
    assert '### Answer A (status: answered; cited: a.md)' in text and '> X\n> because' in text
    assert '(no answer; error:' in text and 'A-All' not in text and 'F-H' not in text

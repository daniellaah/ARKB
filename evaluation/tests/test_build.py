"""Devset construction: substring truth, deterministic hash order, no outcomes consulted."""
import pytest

from evaluation.devset.build import exact_tasks, term_key


def corpus(tmp_path):
    root = tmp_path / 'corpus'
    root.mkdir()
    texts = {'a.md': '# A\n\nQuantum lattice studies mention magnesium twice: magnesium.',
             'b.md': '# B\n\nMagnesium and lattice appear here; quantum too.',
             'c.md': '# C\n\nOnly lattice appears in this one, with tokens galore.',
             'd.md': '# D\n\nNothing shared except lattice and tokens galore again.'}
    for name, text in texts.items():
        (root / name).write_text(text)
    return root


def test_exact_tasks_truth_is_substring_truth_and_deterministic(tmp_path):
    root = corpus(tmp_path)
    strata = {'df_2_5': (2, 5, 2)}
    tasks = exact_tasks(root, 'fixture', scope='fixture-scope', strata=strata, phrase=(2, 10, 1))
    assert len(tasks) == 3 and all(t['scope'] == 'fixture-scope' and 'optional' not in t for t in tasks)
    for task in tasks:
        term = task['term']
        expected = sorted(n for n in ('a.md', 'b.md', 'c.md', 'd.md') if term in (root / n).read_text().split('\n', 2)[2])
        assert task['labels']['expected_sources'] == expected and 2 <= len(expected) <= 10
    again = exact_tasks(root, 'fixture', strata=strata, phrase=(2, 10, 1))
    assert [t['term'] for t in tasks] == [t['term'] for t in again] and again[0]['scope'] == 'fixture'
    assert exact_tasks(root, 'fixture', optional=True, strata=strata, phrase=(2, 10, 1))[0]['optional'] is True
    assert term_key('fixture', 'lattice') != term_key('other', 'lattice')
    with pytest.raises(ValueError, match='Insufficient'):
        exact_tasks(root, 'fixture', strata={'df_2_5': (2, 5, 50)}, phrase=(2, 10, 0))

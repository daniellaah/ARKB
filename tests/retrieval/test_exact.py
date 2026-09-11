import subprocess
from unittest.mock import Mock

import pytest

from arkb.knowledge.documents import DocumentAccess, load_notes
from arkb.retrieval.exact import ExactRetriever


@pytest.fixture
def exact(tmp_path):
    return ExactRetriever(DocumentAccess(tmp_path, vault_id='v'))


def test_exact_empty_scope_and_invalid_pattern(exact):
    assert exact.search('foo').results == ()
    with pytest.raises(ValueError, match='regex parse error'):
        exact.search('[', regex=True)
    with pytest.raises(ValueError):
        exact.search('foo', filters={'unsupported': 'x'})


@pytest.mark.parametrize('body, query, regex, expected', [
    ('éø foo foo', 'foo', False, [(3, 6), (7, 10)]),
    ('\ufefféø foo', 'foo', False, [(4, 7)]),
    ('first\nphrase\nlast', 'first\nphrase', False, [(0, 12)]),
    ('first\nphrase\nlast', r'first\nphrase', True, [(0, 12)]),
    ('foo\nfoo', '^', True, [(0, 0), (4, 4)]),
    ('a+b aab', 'a+b', False, [(0, 3)]),
    ('a+b aab', 'a+b', True, [(4, 7)]),
    ('--example', '--example', False, [(0, 9)]),
    ('a\0foo', 'foo', False, [(2, 5)]),
])
def test_exact_preserves_verbatim_character_coordinates(tmp_path, exact, body, query, regex, expected):
    (tmp_path / 'a.md').write_text(body, encoding='utf-8')
    hits = exact.search(query, regex=regex).results
    assert [(h.start_char, h.end_char) for h in hits] == expected
    note = load_notes(tmp_path)[0]
    for hit in hits:
        assert hit.content == note.content[hit.start_char:hit.end_char]
        assert hit.chunk_id is None and hit.score is None


def test_exact_filters_before_reading_and_stops_at_limit(tmp_path, exact):
    (tmp_path / 'a.md').write_text('word word word', encoding='utf-8')
    (tmp_path / 'b.md').write_bytes(b'\xff')
    assert len(exact.search('word', top_k=1).results) == 1
    assert len(exact.search('word', filters={'source': 'a.md'}).results) == 3
    with pytest.raises(UnicodeDecodeError):
        exact.search('missing')


def test_source_matching_counts_each_document_once_and_preserves_order(tmp_path, exact):
    for source in ('z.md', 'aa.md', 'empty.md'):
        (tmp_path / source).write_text('', encoding='utf-8')
    assert [r.source for r in exact.search('.md', target='source', top_k=2).results] == ['aa.md', 'empty.md']
    hit, = exact.search('a', target='source').results
    assert hit.source == 'aa.md' and hit.content == ''
    assert hit.start_char is None and hit.end_char is None


def test_rg_is_called_with_fixed_argv_and_normalized_files(tmp_path, exact, monkeypatch):
    (tmp_path / 'a.md').write_text('# Title\n\nbody', encoding='utf-8')
    from arkb.retrieval.exact import _run_rg
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        from pathlib import Path
        assert [p.read_text() for p in Path(command[-1]).iterdir()] == ['body']
        return _run_rg(command, **kwargs)
    monkeypatch.setattr('arkb.retrieval.exact._run_rg', run)
    query = '$(touch forbidden); --files'
    assert exact.search(query, case_sensitive=False).results == ()
    assert len(calls) == 1
    command = calls[0]
    assert command[0] == 'rg' and command[command.index('-e') + 1] == query
    assert '--fixed-strings' in command and '--no-config' in command
    assert not (tmp_path / 'forbidden').exists()


def test_rg_failures_are_not_empty_results(tmp_path, exact, monkeypatch):
    (tmp_path / 'a.md').write_text('body', encoding='utf-8')
    run = Mock(return_value=subprocess.CompletedProcess([], 2, stdout='', stderr='I/O failure'))
    monkeypatch.setattr('arkb.retrieval.exact._run_rg', run)
    with pytest.raises(subprocess.CalledProcessError) as raised:
        exact.search('x', regex=True)
    assert raised.value.stderr == 'I/O failure'
    error = FileNotFoundError('rg missing')
    run.side_effect = error
    with pytest.raises(FileNotFoundError) as raised:
        exact.search('x', regex=True)
    assert raised.value is error


@pytest.mark.parametrize('regex,case_sensitive', [(False, True), (True, True), (False, False)])
def test_large_scan_does_not_spawn_per_document(tmp_path, exact, monkeypatch, regex, case_sensitive):
    from arkb.retrieval import exact as module
    original = module.subprocess.Popen
    calls = []
    def spawn(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    for i in range(300):
        (tmp_path / f'{i:04}.md').write_text('haystack')
    (tmp_path / 'z.md').write_text('needle needle')
    assert [h.start_char for h in exact.search('needle', regex=regex, case_sensitive=case_sensitive).results] == [0, 7]
    assert len(calls) <= 1


def test_scan_timeout_and_cancellation_are_explicit(exact):
    from threading import Event
    from arkb.retrieval.exact import ExactCancelled, ExactTimeout
    with pytest.raises(ExactTimeout):
        exact.search('x', timeout=0)
    cancel = Event(); cancel.set()
    with pytest.raises(ExactCancelled):
        exact.search('x', cancel=cancel)


def test_rg_patterns_never_match_across_documents(tmp_path, exact):
    (tmp_path / 'a.md').write_text('first')
    (tmp_path / 'b.md').write_text('second')
    assert not exact.search(r'first\s+second', regex=True).results


@pytest.mark.parametrize('cancelled', [False, True])
def test_running_subprocess_is_reaped_on_timeout_or_cancel(cancelled):
    import sys
    from threading import Event, Timer
    from time import monotonic
    from arkb.retrieval.exact import _run_rg, ExactCancelled, ExactTimeout
    cancel = Event()
    timer = Timer(.1, cancel.set)
    if cancelled:
        timer.start()
    started = monotonic()
    try:
        with pytest.raises(ExactCancelled if cancelled else ExactTimeout):
            _run_rg([sys.executable, '-c', 'import time; time.sleep(60)'],
                    deadline=started + (10 if cancelled else .1), cancel=cancel)
    finally:
        timer.cancel()
    assert monotonic() - started < 5


def test_unicode_case_folding_still_uses_rg(tmp_path, exact):
    (tmp_path / 'a.md').write_text('CAFÉ café Kelvin Kelvin')
    assert [h.content for h in exact.search('café', case_sensitive=False).results] == ['CAFÉ', 'café']
    assert [h.content for h in exact.search('kelvin', case_sensitive=False).results] == ['Kelvin', 'Kelvin']


def test_byte_patterns_cannot_return_partial_unicode_characters(tmp_path, exact):
    (tmp_path / 'a.md').write_text('café', encoding='utf-8')
    with pytest.raises(ValueError, match='complete Unicode'):
        exact.search('(?-u:.)', regex=True)

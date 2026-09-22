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
    assert len(calls) <= 2  # Filename discovery, then one bounded span batch.


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


def test_prepared_matching_reuses_text_and_only_reloads_changed_sources(tmp_path, exact, monkeypatch):
    import os
    from arkb.retrieval import text_cache
    for name in ('a.md', 'b.md'):
        (tmp_path / name).write_text('# Title\n\nold text')
    load = Mock(wraps=text_cache._load_note)
    monkeypatch.setattr(text_cache, '_load_note', load)
    assert exact.prepare()['documents'] == 2
    assert load.call_count == 2
    first = exact.search('old', regex=True).results
    cache_files = {p: p.stat().st_mtime_ns for p in exact._cache.directory.joinpath('content').iterdir()}
    load.reset_mock()
    for options in ({}, {'regex': True}, {'case_sensitive': False}):
        assert exact.search('old', **options).results == first
    load.assert_not_called()
    assert all(p.stat().st_mtime_ns == mtime for p, mtime in cache_files.items())
    # A same-size edit with restored mtime still changes ctime and must invalidate.
    path = tmp_path / 'b.md'
    stat = path.stat()
    path.write_text('# Title\n\nnew text')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert [h.source for h in exact.search('old', regex=True).results] == ['a.md']
    load.assert_called_once_with(path, source='b.md')
    new, = exact.search('new').results
    assert new.metadata['document_revision'] != first[1].metadata['document_revision']
    assert exact.documents.read(new.source_id, start_char=new.start_char,
                                end_char=new.end_char).content == new.content


def test_cached_regex_scope_order_and_filters_follow_insert_delete_and_rename(tmp_path, exact):
    (tmp_path / 'z.md').write_text('needle')
    exact.prepare()
    (tmp_path / 'a.md').write_text('needle')
    for options in ({'regex': True}, {'case_sensitive': False}):
        assert [h.source for h in exact.search('needle', **options).results] == ['a.md', 'z.md']
        assert [h.source for h in exact.search('needle', filters={'source': 'z.md'}, **options).results] == ['z.md']
    assert [h.source for h in exact.search('.md', target='source', regex=True).results] == ['a.md', 'z.md']
    (tmp_path / 'a.md').unlink()
    (tmp_path / 'z.md').rename(tmp_path / 'b.md')
    assert [h.source for h in exact.search('needle', regex=True).results] == ['b.md']
    assert [h.source for h in exact.search('.md', target='source', regex=True).results] == ['b.md']
    assert exact.search('needle', regex=True, filters={'source': 'z.md'}).results == ()


def test_cached_sources_cannot_escape_scope_after_symlink_replacement(tmp_path):
    root = tmp_path / 'notes'; root.mkdir()
    (root / 'a.md').write_text('original')
    outside = tmp_path / 'private.md'; outside.write_text('private')
    with ExactRetriever(DocumentAccess(root, vault_id='v')) as tool:
        tool.prepare()
        (root / 'a.md').unlink()
        (root / 'a.md').symlink_to(outside)
        assert tool.search('original', regex=True).results == ()
        assert tool.search('private', case_sensitive=False).results == ()
        assert tool.search('private', filters={'source': 'a.md'}).results == ()


def test_preparation_interruption_is_resumable_and_does_not_hide_bad_input(tmp_path, exact, monkeypatch):
    from threading import Event
    from arkb.retrieval import text_cache
    from arkb.retrieval.exact import ExactCancelled
    for name in ('a.md', 'b.md'):
        (tmp_path / name).write_text('needle')
    cancel = Event()
    load = text_cache._load_note
    def interrupted(path, *, source=None):
        note = load(path, source=source)
        if path.name == 'b.md':
            cancel.set()
        return note
    monkeypatch.setattr(text_cache, '_load_note', interrupted)
    with pytest.raises(ExactCancelled):
        exact.prepare(cancel=cancel)
    assert len(exact._cache._entries) == 1
    monkeypatch.setattr(text_cache, '_load_note', load)
    assert exact.prepare()['documents'] == 2
    (tmp_path / 'b.md').write_bytes(b'\xff')
    with pytest.raises(UnicodeDecodeError):
        exact.search('needle', regex=True)


def test_prepared_cache_cleanup_is_explicit(tmp_path, exact):
    (tmp_path / 'a.md').write_text('needle')
    exact.prepare()
    root = exact._cache.directory
    assert root.exists()
    exact.close()
    exact.close()
    assert not root.exists()
    with pytest.raises(RuntimeError, match='closed'):
        exact.search('needle')


@pytest.mark.parametrize('source', ['bad\\source.md', 'C:note.md'])
def test_cache_preserves_document_source_validation(tmp_path, exact, source):
    (tmp_path / source).write_text('needle')
    with pytest.raises(ValueError, match='canonical'):
        exact.prepare()


def test_cache_detects_partial_writes(tmp_path, exact, monkeypatch):
    (tmp_path / 'a.md').write_text('needle')
    exact._cache.directory  # Allocate the private store before injecting failure.
    write = exact._cache._bodies.write
    monkeypatch.setattr(exact._cache._bodies, 'write', lambda body: write(body[:1]))
    with pytest.raises(OSError, match='Incomplete'):
        exact.prepare()
    assert not exact._cache._entries


def test_waiting_for_shared_cache_obeys_timeout(exact):
    from threading import Event, Thread
    from time import monotonic
    from arkb.retrieval.exact import ExactTimeout
    entered, release = Event(), Event()
    def busy():
        with exact._lock:
            entered.set()
            release.wait(5)
    thread = Thread(target=busy)
    thread.start()
    assert entered.wait(2)
    start = monotonic()
    try:
        with pytest.raises(ExactTimeout):
            exact.search('needle', timeout=.03)
        assert monotonic()-start < .5
    finally:
        release.set()
        thread.join(2)


def test_unique_sources_lists_each_document_once_and_reports_truncation(tmp_path, exact):
    (tmp_path / 'a.md').write_text('word word word', encoding='utf-8')
    (tmp_path / 'b.md').write_text('word', encoding='utf-8')
    (tmp_path / 'c.md').write_text('word\nword', encoding='utf-8')
    (tmp_path / 'd.md').write_text('nothing here', encoding='utf-8')
    unique = exact.search('word', unique_sources=True, top_k=2)
    assert [r.source for r in unique.results] == ['a.md', 'b.md'] and unique.truncated is True
    assert [(r.start_char, r.end_char) for r in unique.results] == [(0, 4), (0, 4)]
    complete = exact.search('word', unique_sources=True, top_k=3)
    assert [r.source for r in complete.results] == ['a.md', 'b.md', 'c.md'] and complete.truncated is False
    assert exact.search('word', unique_sources=True, top_k=50).truncated is False
    occurrences = exact.search('word', top_k=2)
    assert [r.source for r in occurrences.results] == ['a.md', 'a.md'] and occurrences.truncated is True
    assert exact.search('word', top_k=6).truncated is False
    assert exact.search('word', top_k=5).truncated is True
    absent = exact.search('absent', unique_sources=True)
    assert absent.results == () and absent.truncated is False
    regex = exact.search('WORD', case_sensitive=False, unique_sources=True, top_k=1)
    assert [r.source for r in regex.results] == ['a.md'] and regex.truncated is True
    filtered = exact.search('word', filters={'source': 'c.md'}, unique_sources=True, top_k=1)
    assert [r.source for r in filtered.results] == ['c.md'] and filtered.truncated is False
    with pytest.raises(ValueError):
        exact.search('word', unique_sources='yes')


def test_same_filename_in_two_folders_keeps_separate_cached_bodies(tmp_path):
    for folder, body in (('one', 'needle in one'), ('two', 'other text, needle in two')):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'note.md').write_text(f'# Note\n\n{body}', encoding='utf-8')
    (tmp_path / 'note.md').write_text('# Note\n\nneedle at the root', encoding='utf-8')
    with ExactRetriever(DocumentAccess(tmp_path, vault_id='v')) as tool:
        assert tool.prepare()['documents'] == 3
        for options in ({}, {'regex': True}, {'case_sensitive': False}):
            hits = tool.search('needle', top_k=10, **options).results
            assert [(hit.source, hit.start_char) for hit in hits] == [
                ('note.md', 0), ('one/note.md', 0), ('two/note.md', 12)]
        # Every hit still expands to its own live document, not a same-named sibling.
        for hit in tool.search('needle', top_k=10).results:
            document = tool.documents.read(hit.source_id)
            assert document.source == hit.source
            assert document.content[hit.start_char:hit.end_char] == 'needle'
        assert [h.source for h in tool.search('one/note.md', target='source').results] == ['one/note.md']
        assert [h.source for h in tool.search('needle', filters={'source': 'two/note.md'}).results] == ['two/note.md']


def test_cached_bodies_follow_an_edit_to_one_of_two_same_named_notes(tmp_path):
    for folder in ('one', 'two'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'note.md').write_text('# Note\n\nold body', encoding='utf-8')
    with ExactRetriever(DocumentAccess(tmp_path, vault_id='v')) as tool:
        tool.prepare()
        (tmp_path / 'two' / 'note.md').write_text('# Note\n\nnew body', encoding='utf-8')
        assert [h.source for h in tool.search('old body', top_k=10).results] == ['one/note.md']
        assert [h.source for h in tool.search('new body', top_k=10, regex=True).results] == ['two/note.md']


def test_excluded_folders_stay_out_of_exact_matching(tmp_path):
    (tmp_path / 'Attachments').mkdir()
    (tmp_path / 'Attachments' / 'clip.md').write_text('needle', encoding='utf-8')
    (tmp_path / 'kept.md').write_text('needle', encoding='utf-8')
    with ExactRetriever(DocumentAccess(tmp_path, vault_id='v')) as tool:
        assert [h.source for h in tool.search('needle', top_k=10).results] == ['kept.md']
        assert tool.search('needle', filters={'source': 'Attachments/clip.md'}).results == ()

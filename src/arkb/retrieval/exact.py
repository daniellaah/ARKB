"""Bounded live exact matching in normalized body coordinates.

Case-sensitive literals need no subprocess. Rust regex and Unicode case folding
use separate normalized files so patterns cannot bridge sources. The session
cache revalidates live source signatures and only rewrites changed documents.
"""
from collections.abc import Mapping
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
from threading import Event, RLock
from time import monotonic

from arkb.knowledge.documents import DocumentAccess
from arkb.knowledge.models import _document_id
from arkb.retrieval.models import SearchResponse, SearchResult, validate_request
from arkb.retrieval.text_cache import TextCache


class ExactTimeout(TimeoutError):
    """The bounded exact operation expired; no partial success is implied."""


class ExactCancelled(RuntimeError):
    """The caller cancelled exact matching."""


class ExactPatternError(ValueError):
    """An invalid rg pattern or one addressing partial Unicode characters."""


def _check(deadline, cancel):
    if cancel is not None and cancel.is_set():
        raise ExactCancelled('Exact matching was cancelled.')
    if monotonic() >= deadline:
        raise ExactTimeout('Exact matching exceeded its deadline.')


def _run_rg(command, *, deadline, cancel):
    _check(deadline, cancel)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          encoding='utf-8') as process:
        try:
            while True:
                _check(deadline, cancel)
                try:
                    stdout, stderr = process.communicate(timeout=min(.05, max(.001, deadline-monotonic())))
                    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.kill()
            process.communicate()
            raise


def _literal_matches(text, query):
    start = 0
    while (position := text.find(query, start)) >= 0:
        yield position, position + len(query)
        start = position + len(query)


def _batched_matches(entries, query, *, cache, filtered, target, regex, case_sensitive, deadline, cancel):
    indexed = {entry.path.name: entry for entry in entries}
    path = cache.rg_path(entries, target=target, filtered=filtered,
                         check=lambda: _check(deadline, cancel))
    command = ['rg', '--no-config', '--text', '--multiline', '--encoding', 'none',
               '--threads', '4', '--no-ignore', '--hidden',
               '--case-sensitive' if case_sensitive else '--ignore-case']
    if not regex:
        command.append('--fixed-strings')

    def run(options):
        completed = _run_rg([*command, *options], deadline=deadline, cancel=cancel)
        if completed.returncode not in (0, 1):
            if completed.returncode == 2 and 'regex parse error' in completed.stderr:
                raise ExactPatternError(completed.stderr.strip())
            raise subprocess.CalledProcessError(completed.returncode, command,
                                                output=completed.stdout, stderr=completed.stderr)
        return completed.stdout

    # Discover filenames first. Broad patterns must not serialize matching text
    # from the whole corpus just to return a handful of early occurrences.
    names = run(['--files-with-matches', '--null', '-e', query, '--', str(path)])
    selected = sorted((indexed[Path(name).name] for name in names.split('\0') if name),
                      key=lambda entry: entry.source)
    # Small batches keep argv bounded. The consumer closes this generator once
    # top_k is reached, so later source batches are never materialized as JSON.
    for start in range(0, len(selected), 32):
        paths = [str(cache.directory / target / entry.path.name) for entry in selected[start:start+32]]
        output = run(['--json', '-e', query, '--', *paths])
        matches = []
        for line in output.splitlines():
            _check(deadline, cancel)
            event = json.loads(line)
            if event['type'] == 'match':
                data = event['data']
                entry = indexed[Path(data['path']['text']).name]
                matches.append((entry.source, data['absolute_offset'], entry, data))
        # Stable source ordering is independent of cache names and rg threads.
        for _, _, entry, data in sorted(matches, key=lambda item: item[:2]):
            _check(deadline, cancel)
            body = entry.body() if target == 'content' else entry.source.encode('utf-8')
            for match in data['submatches']:
                start = data['absolute_offset'] + match['start']
                end = data['absolute_offset'] + match['end']
                try:
                    span = len(body[:start].decode('utf-8')), len(body[:end].decode('utf-8'))
                except UnicodeDecodeError as error:
                    raise ExactPatternError('Patterns must match complete Unicode characters.') from error
                yield entry, *span


class ExactRetriever:
    """Find literal strings or explicit patterns, ordered by source then position."""

    def __init__(self, documents: DocumentAccess):
        self.documents = documents
        self._cache = TextCache(documents)
        self._lock = RLock()

    @contextmanager
    def _locked(self, deadline, cancel):
        # Contention on a shared cache is part of the caller's elapsed budget.
        while not self._lock.acquire(timeout=min(.05, max(0, deadline-monotonic()))):
            _check(deadline, cancel)
        try:
            _check(deadline, cancel)
            yield
        finally:
            self._lock.release()

    def prepare(self, *, timeout: float | None = None, cancel: Event | None = None):
        """Explicitly prepare reusable text outside query timing; no model/index build.

        Preparation is optional for small/live scopes. Large services should
        call it once before accepting queries and report its cost separately.
        """
        if timeout is not None and (type(timeout) not in (int, float) or not 0 <= timeout < float('inf')):
            raise ValueError('timeout must be finite and nonnegative.')
        deadline = float('inf') if timeout is None else monotonic() + timeout
        with self._locked(deadline, cancel):
            _check(deadline, cancel)
            entries = list(self._cache.scan(check=lambda: _check(deadline, cancel)))
            self._cache.rg_path(entries, target='content', filtered=False,
                                check=lambda: _check(deadline, cancel))
            _check(deadline, cancel)
            return {'documents': len(entries), 'bytes': sum(e.size for e in entries)}

    def close(self):
        with self._lock:
            self._cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def search(self, query: str, *, target: str = 'content', regex: bool = False,
               case_sensitive: bool = True, top_k: int = 5,
               filters: Mapping[str, str] | None = None, timeout: float = 30,
               cancel: Event | None = None, unique_sources: bool = False) -> SearchResponse:
        """Occurrences in source order, or one hit per source with unique_sources.

        The response reports truncated=True when at least one further eligible
        match existed beyond top_k, so a caller can enumerate completely by
        raising top_k or narrowing the pattern instead of guessing.
        """
        filters = validate_request(query, top_k, filters)
        if target not in ('content', 'source'):
            raise ValueError('target must be content or source.')
        if type(regex) is not bool or type(case_sensitive) is not bool or type(unique_sources) is not bool:
            raise ValueError('regex, case_sensitive and unique_sources must be booleans.')
        if '\x00' in query:
            raise ExactPatternError('query must not contain a NUL character.')
        if type(timeout) not in (int, float) or not 0 <= timeout < float('inf'):
            raise ValueError('timeout must be finite and nonnegative.')
        deadline = monotonic() + timeout
        _check(deadline, cancel)
        with self._locked(deadline, cancel):
            return self._search(query, target=target, regex=regex, case_sensitive=case_sensitive,
                                top_k=top_k, source=filters.get('source'), deadline=deadline, cancel=cancel,
                                unique_sources=unique_sources)

    def _search(self, query, *, target, regex, case_sensitive, top_k, source, deadline, cancel, unique_sources=False):
        entries = self._cache.scan(source=source, check=lambda: _check(deadline, cancel))

        def literals():
            for entry in entries:
                _check(deadline, cancel)
                text = entry.body().decode('utf-8') if target == 'content' else entry.source
                for start, end in _literal_matches(text, query):
                    _check(deadline, cancel)
                    yield entry, start, end

        matches = (literals() if not regex and case_sensitive else _batched_matches(
            list(entries), query, cache=self._cache, filtered=source is not None,
            target=target, regex=regex, case_sensitive=case_sensitive,
            deadline=deadline, cancel=cancel))
        results, sources = [], set()
        notes = {}
        truncated = False
        one_per_source = target == 'source' or unique_sources
        try:
            for entry, start, end in matches:
                if one_per_source and entry.source in sources:
                    continue
                if len(results) == top_k:
                    # One further eligible match proves the list is incomplete.
                    truncated = True
                    break
                sources.add(entry.source)
                if entry.source not in notes:
                    note = entry.note()
                    notes[entry.source] = (note, note.document_revision)
                note, revision = notes[entry.source]
                results.append(SearchResult(
                    source_id=_document_id(self.documents.vault_id, entry.source), source=entry.source, method='exact',
                    content=note.content[start:end] if target == 'content' else note.content,
                    start_char=start if target == 'content' else None,
                    end_char=end if target == 'content' else None,
                    metadata={'title': note.title, 'document_revision': revision},
                ))
        finally:
            matches.close()
        _check(deadline, cancel)
        return SearchResponse(query=query, method='exact', results=tuple(results), truncated=truncated)

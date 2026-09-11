"""Bounded live exact matching in normalized body coordinates.

Case-sensitive literals need no subprocess. Rust regex and Unicode case folding
remain ripgrep operations, batched over separate normalized files so patterns
cannot bridge sources. Temporary names never contain model-supplied paths.
"""
from collections.abc import Iterator, Mapping
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic

from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval.models import SearchResponse, SearchResult, validate_request


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


def _batched_matches(records, query, *, target, regex, case_sensitive, deadline, cancel):
    with TemporaryDirectory(prefix='arkb-exact-') as directory:
        indexed = []
        for record in records:
            _check(deadline, cancel)
            text = record.chunk.content if target == 'content' else record.chunk.source
            Path(directory, f'{len(indexed):09d}').write_text(text, encoding='utf-8')
            indexed.append((record, text.encode('utf-8')))
        # Still validate regex when the corpus is empty.
        if not indexed:
            Path(directory, 'empty').write_text('')
        command = ['rg', '--no-config', '--json', '--text', '--multiline', '--encoding', 'none',
                   '--sort', 'path', '--no-ignore', '--hidden',
                   '--case-sensitive' if case_sensitive else '--ignore-case']
        if not regex:
            command.append('--fixed-strings')
        completed = _run_rg([*command, '-e', query, '--', directory], deadline=deadline, cancel=cancel)
        if completed.returncode not in (0, 1):
            if completed.returncode == 2 and 'regex parse error' in completed.stderr:
                raise ExactPatternError(completed.stderr.strip())
            raise subprocess.CalledProcessError(completed.returncode, command,
                                                output=completed.stdout, stderr=completed.stderr)
        for line in completed.stdout.splitlines():
            _check(deadline, cancel)
            event = json.loads(line)
            if event['type'] != 'match':
                continue
            data = event['data']
            record, body = indexed[int(Path(data['path']['text']).name)]
            for match in data['submatches']:
                start = data['absolute_offset'] + match['start']
                end = data['absolute_offset'] + match['end']
                try:
                    span = len(body[:start].decode('utf-8')), len(body[:end].decode('utf-8'))
                except UnicodeDecodeError as error:
                    raise ExactPatternError('Patterns must match complete Unicode characters.') from error
                yield record, *span


class ExactRetriever:
    """Find literal strings or explicit patterns, ordered by source then position."""

    def __init__(self, documents: DocumentAccess):
        self.documents = documents

    def search(self, query: str, *, target: str = 'content', regex: bool = False,
               case_sensitive: bool = True, top_k: int = 5,
               filters: Mapping[str, str] | None = None, timeout: float = 30,
               cancel: Event | None = None) -> SearchResponse:
        filters = validate_request(query, top_k, filters)
        if target not in ('content', 'source'):
            raise ValueError('target must be content or source.')
        if type(regex) is not bool or type(case_sensitive) is not bool:
            raise ValueError('regex and case_sensitive must be booleans.')
        if '\x00' in query:
            raise ExactPatternError('query must not contain a NUL character.')
        if type(timeout) not in (int, float) or not 0 <= timeout < float('inf'):
            raise ValueError('timeout must be finite and nonnegative.')
        deadline = monotonic() + timeout
        _check(deadline, cancel)
        records = self.documents.records(source=filters.get('source'))

        def literals():
            for record in records:
                _check(deadline, cancel)
                text = record.chunk.content if target == 'content' else record.chunk.source
                for start, end in _literal_matches(text, query):
                    _check(deadline, cancel)
                    yield record, start, end

        matches = (literals() if not regex and case_sensitive else _batched_matches(
            records, query, target=target, regex=regex, case_sensitive=case_sensitive,
            deadline=deadline, cancel=cancel))
        results, sources = [], set()
        try:
            for record, start, end in matches:
                chunk = record.chunk
                if target == 'source' and chunk.source in sources:
                    continue
                sources.add(chunk.source)
                results.append(SearchResult(
                    source_id=record.document_id, source=chunk.source, method='exact',
                    content=chunk.content[start:end] if target == 'content' else chunk.content,
                    start_char=start if target == 'content' else None,
                    end_char=end if target == 'content' else None,
                    metadata={'title': chunk.title, 'document_revision': record.document_revision},
                ))
                if len(results) == top_k:
                    break
        finally:
            matches.close()
        _check(deadline, cancel)
        return SearchResponse(query=query, method='exact', results=tuple(results))

"""Session-owned normalized files; source signatures are checked on every scan.

The cache is disposable and contains no search results. Bodies stay on disk,
not in an unbounded Python dictionary. It never replaces the live source scope.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from arkb.knowledge.documents import DocumentAccess, _load_note
from arkb.knowledge.models import Note


def _signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@dataclass(frozen=True)
class CachedText:
    source: str
    title: str
    signature: tuple[int, ...]
    path: Path
    size: int
    descriptor: int
    offset: int

    def body(self):
        value = os.pread(self.descriptor, self.size, self.offset)
        if len(value) != self.size:
            raise OSError('Incomplete normalized text cache read.')
        return value

    def note(self):
        return Note(self.title, self.body().decode('utf-8'), self.source)


class TextCache:
    def __init__(self, documents: DocumentAccess):
        self.documents = documents
        self._temporary = None
        self._entries = {}
        self._serial = 0
        self._closed = False
        self._bodies = None
        self._materialized = {'content': set(), 'source': set()}

    @property
    def directory(self):
        if self._closed:
            raise RuntimeError('Text cache is closed.')
        if self._temporary is None:
            self._temporary = TemporaryDirectory(prefix='arkb-exact-')
            root = Path(self._temporary.name)
            for name in ('content', 'source', 'empty'):
                (root / name).mkdir()
            self._bodies = (root / 'bodies').open('w+b', buffering=0)
        return Path(self._temporary.name)

    def scan(self, *, source=None, check):
        if self._closed:
            raise RuntimeError('Text cache is closed.')
        seen = set()
        for path in self.documents._paths(source):
            check()
            seen.add(path.name)
            signature = _signature(path)
            entry = self._entries.get(path.name)
            if entry is None or entry.signature != signature:
                note = _load_note(path)
                # Loading a flat filename still has to enforce the canonical
                # source contract previously checked by ChunkRecord creation.
                if '\\' in note.source or re.match(r'^[A-Za-z]:', note.source):
                    raise ValueError('source must be a canonical vault-relative POSIX path.')
                if _signature(path) != signature:
                    raise ValueError('Document changed while preparing exact matching; retry the call.')
                check()
                if entry is None:
                    cached = self.directory / 'content' / f'{self._serial:09d}'
                    self._serial += 1
                else:
                    cached = entry.path
                body = note.content.encode('utf-8')
                # One packed file avoids opening 100,000 cached files per
                # literal query. Reuse a slot when an edit fits its old size.
                offset = entry.offset if entry is not None and len(body) <= entry.size else self._bodies.seek(0, 2)
                self._bodies.seek(offset)
                if self._bodies.write(body) != len(body):
                    raise OSError('Incomplete normalized text cache write.')
                self._materialized['content'].discard(cached.name)
                entry = CachedText(note.source, note.title, signature, cached, len(body),
                                   self._bodies.fileno(), offset)
                self._entries[path.name] = entry
            yield entry
        # Only a complete, unfiltered scan can prove that a cached source left
        # the scope. A timed-out/early-terminated scan remains safely resumable.
        if source is None:
            for name in self._entries.keys() - seen:
                entry = self._entries.pop(name)
                entry.path.unlink(missing_ok=True)
                (self.directory / 'source' / entry.path.name).unlink(missing_ok=True)
                for names in self._materialized.values():
                    names.discard(entry.path.name)

    def rg_path(self, entries, *, target, filtered, check):
        root = self.directory
        if not entries:
            # rg must still validate invalid patterns in an empty scope.
            (root / 'empty' / 'empty').touch(exist_ok=True)
            return root / 'empty'
        for entry in entries:
            check()
            if entry.path.name not in self._materialized[target]:
                path = root / target / entry.path.name
                path.write_bytes(entry.body() if target == 'content' else entry.source.encode('utf-8'))
                self._materialized[target].add(entry.path.name)
        return (root / target / entries[0].path.name) if filtered else root / target

    def close(self):
        if self._bodies is not None:
            self._bodies.close()
            self._bodies = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self._entries.clear()
        self._closed = True

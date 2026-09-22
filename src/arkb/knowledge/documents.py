"""Load and access current Markdown documents in a nested vault scope.

A source is a vault-relative POSIX path ("04-Areas/Career/note.md"); a flat
knowledge base is the special case where every source is a bare filename.
"""

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatchcase
import os
from pathlib import Path, PurePosixPath
import re

from arkb.knowledge.chunking import _markdown_blocks, _sections, whole_note_chunks
from arkb.knowledge.models import (
    ChunkRecord, Note, _document_id, _require_digest, _require_text, is_canonical_source,
)


# Directory globs excluded from every scan. Dot-directories hold vault state
# (.obsidian, .trash, .git) and the two named folders hold binary attachments
# and drawings, not prose. Callers pass their own tuple to widen or narrow this.
DEFAULT_EXCLUDES: tuple[str, ...] = ('.*', 'Attachments', 'Excalidraw')


class DocumentNotFound(LookupError):
    """A live document or requested section no longer resolves in this scope."""


class FrontmatterError(ValueError):
    """A delimited frontmatter block outside the YAML subset parsed here."""


@dataclass(frozen=True)
class DocumentSlice:
    """Current document text in body coordinates, without an indexed chunk identity."""

    document_id: str
    document_revision: str
    source: str
    title: str
    content: str
    start_char: int
    end_char: int
    section_id: str | None = None
    heading_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkippedNote:
    """One file a scan could not fully use, with the reason to report."""

    source: str
    reason: str


@dataclass(frozen=True)
class VaultScan:
    """Loaded notes plus the files a whole-vault scan could not fully use."""

    notes: tuple[Note, ...] = ()
    unreadable: tuple[SkippedNote, ...] = ()
    unparsed_metadata: tuple[SkippedNote, ...] = ()


_FRONTMATTER_KEY = re.compile(r'^([^\s:#][^:]*):(.*)$')


def _scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
        value = value[1:-1]
    return value


def _split_frontmatter(text: str) -> tuple[list[str] | None, str]:
    """Separate a leading "---" delimited block from the body.

    Return (block lines, body), or (None, text) when the file does not open
    with a delimiter line or the block is never closed: an unterminated block
    is ordinary Markdown, such as a note starting with a horizontal rule.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != '---':
        return None, text
    for index in range(1, len(lines)):
        if lines[index].strip() in ('---', '...'):
            return lines[1:index], ''.join(lines[index + 1:])
    return None, text


def _parse_frontmatter(lines: Sequence[str]) -> dict:
    """Parse the Obsidian subset of YAML: scalars, block lists and inline lists.

    Handles "key: value", "key:" followed by "- item" lines, "key: [a, b]",
    blank lines and # comments. Surrounding single or double quotes are
    removed. A key with neither a value nor items keeps the empty string.
    Anything else -- nested mappings, multi-line scalars, anchors -- raises
    FrontmatterError so the caller can keep the body and report the note.
    """
    metadata: dict = {}
    key = None
    for line in lines:
        raw = line.rstrip('\n').rstrip('\r')
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if stripped.startswith('- ') or stripped == '-':
            if key is None:
                raise FrontmatterError('A list item precedes its key.')
            items = metadata[key] if isinstance(metadata[key], list) else []
            metadata[key] = [*items, _scalar(stripped[1:])]
            continue
        if raw != raw.lstrip():
            raise FrontmatterError('Nested mappings are not supported.')
        match = _FRONTMATTER_KEY.match(raw)
        if match is None:
            raise FrontmatterError(f'Not a "key: value" line: {raw[:40]!r}.')
        key, value = match.group(1).strip(), match.group(2).strip()
        if not key:
            raise FrontmatterError('A frontmatter key must not be blank.')
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            metadata[key] = [_scalar(item) for item in inner.split(',')] if inner else []
        else:
            metadata[key] = _scalar(value)
    return metadata


def _read_note(path: Path, source: str) -> tuple[Note, str | None]:
    """Load one note and report an unparsable frontmatter block instead of failing.

    A delimited block is always removed from the body, even when this YAML
    subset cannot parse it; the note then carries empty metadata.
    """
    text = path.read_text(encoding='utf-8')
    block, text = _split_frontmatter(text)
    metadata, error = {}, None
    if block is not None:
        try:
            metadata = _parse_frontmatter(block)
        except FrontmatterError as failure:
            error = str(failure)
    lines = text.splitlines(keepends=True)
    title = PurePosixPath(source).stem
    content = text
    for index, line in enumerate(lines):
        if line.startswith('# '):
            title = line.removeprefix('# ').strip()
            content = ''.join(lines[:index] + lines[index + 1:])
            break
    return Note(title=title, content=content.strip(), source=source, metadata=metadata), error


def _load_note(path: Path, *, source: str | None = None) -> Note:
    """Read one note, defaulting a flat caller's source to the filename."""
    return _read_note(path, path.name if source is None else source)[0]


# A tag starts a word, holds at least one letter, and nests with "/". The
# word boundary keeps "## Heading" and the fragment of "http://host#anchor"
# out; a purely numeric "#1" is an issue reference, not a tag.
_INLINE_TAG = re.compile(r'(?:^|(?<=\s))#([\w/-]*[^\W\d_][\w/-]*)', re.MULTILINE)


def note_tags(note: Note) -> tuple[str, ...]:
    """Collect a note's tags: frontmatter "tags"/"tag" plus inline #tags in the body.

    Frontmatter values may be a list or one string holding several tags
    separated by commas or spaces, with or without a leading "#". Tags keep the
    spelling they were written in, in frontmatter-then-body order, without
    duplicates.
    """
    found: list[str] = []
    for key in ('tags', 'tag'):
        value = note.metadata.get(key)
        for item in value if isinstance(value, list) else [value] if isinstance(value, str) else []:
            if isinstance(item, str):
                found.extend(part.lstrip('#') for part in re.split(r'[,\s]+', item) if part.strip('#'))
    found.extend(match.group(1) for match in _INLINE_TAG.finditer(note.content))
    return tuple(dict.fromkeys(tag for tag in found if tag))


def _tagged(tags: Sequence[str], wanted: str) -> bool:
    """Match a tag case-insensitively, and match a parent of a nested tag."""
    wanted = wanted.lstrip('#').lower()
    return any(tag.lower() == wanted or tag.lower().startswith(wanted + '/') for tag in tags)


def _instant(value: str, name: str) -> float:
    """Read an ISO date or datetime as a local-time epoch second."""
    _require_text(value, name)
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError as error:
        raise ValueError(f'{name} must be an ISO date such as 2026-01-31, or an ISO datetime.') from error


def _excluded(name: str, relative: str, exclude: Sequence[str]) -> bool:
    """Match a directory by its own name or by its vault-relative path."""
    return any(fnmatchcase(name, pattern) or fnmatchcase(relative, pattern)
               for pattern in exclude)


def note_files(directory: Path, *, exclude: Sequence[str] = DEFAULT_EXCLUDES
               ) -> Iterator[tuple[Path, str]]:
    """Walk the Markdown scope, yielding each file with its vault-relative source.

    Order is depth first with every directory sorted by entry name, so a scan
    is reproducible. Directory symlinks are not followed, which bounds the walk
    on a vault that links to one of its own ancestors; file symlinks are
    followed as before. Loading/indexing follow file symlinks. Live tools and
    evaluation fingerprints additionally exclude paths outside their knowledge
    root. A file whose name begins with "." is still in scope: only
    directories are excluded.
    """
    def visit(root: str, prefix: str) -> Iterator[tuple[Path, str]]:
        with os.scandir(root) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        for entry in entries:
            relative = prefix + entry.name
            if entry.is_dir(follow_symlinks=False):
                if not _excluded(entry.name, relative, exclude):
                    yield from visit(entry.path, relative + '/')
            elif entry.name.endswith('.md') and entry.name != '.md' and entry.is_file():
                yield Path(entry.path), relative

    yield from visit(os.fspath(directory), '')


def load_notes(directory: Path, *, exclude: Sequence[str] = DEFAULT_EXCLUDES) -> list[Note]:
    """Read UTF-8 .md files under directory, in vault-relative source order.

    Strip a leading YAML frontmatter block into Note.metadata. Use the first
    remaining line starting with "# " as the title, or the source stem if there
    is no such line. Remove the title line from the body and strip leading and
    trailing whitespace.

    Filesystem and decoding errors propagate to the caller; scan_notes skips
    and reports them instead.
    """
    return [_load_note(path, source=source)
            for path, source in note_files(directory, exclude=exclude)]


class DocumentAccess:
    """Resolve existing document IDs and read live files without retaining content.

    The directory, vault and exclusions are supplied by the application, never
    a tool call. IDs use the same vault/path namespace as
    ChunkRecord.document_id. File edits are visible on the next call; deleting
    or renaming a file removes its old ID. Symlinks outside the knowledge root
    are excluded from document access.
    """

    def __init__(self, directory: Path, *, vault_id: str,
                 exclude: Sequence[str] = DEFAULT_EXCLUDES):
        _require_text(vault_id, 'vault_id')
        self.directory = Path(directory).resolve()
        self.vault_id = vault_id
        self.exclude = tuple(exclude)

    def _confined(self, path: Path) -> bool:
        return not path.is_symlink() or path.resolve().is_relative_to(self.directory)

    def _paths(self, source: str | None = None) -> Iterator[tuple[Path, str]]:
        if source is not None:
            _require_text(source, 'source')
            # A source selector addresses one canonical vault-relative path.
            with os.scandir(self.directory):
                pass  # Preserve missing-directory and non-directory failures.
            if not is_canonical_source(source) or not source.endswith('.md'):
                return
            parts = PurePosixPath(source).parts
            if any(_excluded(part, '/'.join(parts[:index + 1]), self.exclude)
                   for index, part in enumerate(parts[:-1])):
                return
            path = self.directory.joinpath(*parts)
            if path.resolve().is_relative_to(self.directory) and path.is_file():
                yield path, source
            return
        for path, relative in note_files(self.directory, exclude=self.exclude):
            if self._confined(path):
                yield path, relative

    def list(self, pattern: str | None = None, *, tag: str | None = None,
             modified_after: str | None = None, modified_before: str | None = None,
             limit: int = 50, max_headings: int = 12) -> dict:
        """Summarize notes in source order: source, title, headings, size, tags and mtime.

        pattern filters the vault-relative path case-insensitively, so a folder
        can be selected with "04-Areas/*"; a pattern without glob characters
        matches as a substring, and "*" also matches "/". tag matches a
        frontmatter or inline tag, including a parent of a nested tag.
        modified_after and modified_before bound the file's modification time
        as a half-open local-time range: a note is kept when its mtime is at or
        after modified_after and strictly before modified_before, so a bare
        date includes that whole day only on the after side. Every given filter
        must hold.

        Nothing here is evidence: the listing lets a caller decide what to
        search or read, and reports `truncated` when more notes matched than
        limit.
        """
        if pattern is not None:
            _require_text(pattern, 'pattern')
            pattern = pattern.lower()
            if not any(c in pattern for c in '*?['):
                pattern = f'*{pattern}*'
        if tag is not None:
            _require_text(tag, 'tag')
        after = None if modified_after is None else _instant(modified_after, 'modified_after')
        before = None if modified_before is None else _instant(modified_before, 'modified_before')
        if type(limit) is not int or limit < 1:
            raise ValueError('limit must be a positive integer.')
        notes, total = [], 0
        for path, source in self._paths():
            if pattern is not None and not fnmatchcase(source.lower(), pattern):
                continue
            mtime = path.stat().st_mtime
            if (after is not None and mtime < after) or (before is not None and mtime >= before):
                continue
            # A tag is only known once the note is read, so a tag filter costs
            # one read per candidate; the other filters still decide first.
            note = _load_note(path, source=source) if tag is not None else None
            tags = () if note is None else note_tags(note)
            if tag is not None and not _tagged(tags, tag):
                continue
            total += 1
            if len(notes) >= limit:
                continue
            if note is None:
                note = _load_note(path, source=source)
                tags = note_tags(note)
            headings = ['#' * block.level + ' ' + block.heading
                        for block in _markdown_blocks(note.content) if block.kind == 'heading']
            notes.append({'source': note.source, 'title': note.title, 'chars': len(note.content),
                          'modified': datetime.fromtimestamp(mtime).isoformat(timespec='seconds'),
                          **({'tags': list(tags)} if tags else {}),
                          'headings': headings[:max_headings],
                          **({'more_headings': len(headings) - max_headings} if len(headings) > max_headings else {})})
        return {'notes': notes, 'total': total, 'truncated': total > len(notes)}

    def titles(self, sources: Iterable[str]) -> dict[str, str]:
        """Current titles for known sources, omitting any outside the live scope.

        A caller that has paths from elsewhere -- the link graph, for instance
        -- uses this to name them without reading them as evidence.
        """
        found = {}
        for source in sources:
            for path, relative in self._paths(source):
                try:
                    found[relative] = _load_note(path, source=relative).title
                except (OSError, UnicodeDecodeError):
                    continue
        return found

    def records(self, *, source: str | None = None) -> Iterator[ChunkRecord]:
        """Yield current complete bodies in source order, filtering before I/O.

        These are source slices represented with existing records, not indexed
        chunks; consumers must not advertise their synthetic chunk IDs.
        """
        for path, relative in self._paths(source):
            note = _load_note(path, source=relative)
            yield ChunkRecord.from_note(whole_note_chunks([note])[0], note=note,
                                        vault_id=self.vault_id)

    def read(self, document_id: str | None = None, *, source: str | None = None,
             section_id: str | None = None,
             start_char: int | None = None, end_char: int | None = None) -> DocumentSlice:
        """Read a full body, Markdown section, or end-exclusive character range.

        Supply a document ID or exact vault-relative source path; if both are
        provided, they must identify the same document. Source lookup uses the
        same directory scope, exclusions and symlink rules as ID lookup.
        A missing range endpoint means the corresponding document boundary.
        Sections include their heading and direct body, up to the next heading.
        Section IDs always refer to current Markdown sections. Use an unqualified
        read for a whole document, including hits from whole-note indexes.
        """
        if document_id is None and source is None:
            raise ValueError('Supply document_id or source.')
        if document_id is not None:
            _require_digest(document_id, 'document_id')
        if source is not None:
            _require_text(source, 'source')
        if section_id is not None:
            _require_digest(section_id, 'section_id')
            if start_char is not None or end_char is not None:
                raise ValueError('section_id and character range are mutually exclusive.')
        for name, value in (('start_char', start_char), ('end_char', end_char)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer.')
        if start_char is not None and end_char is not None and start_char > end_char:
            raise ValueError('start_char must not exceed end_char.')

        # Resolve by path identity without loading unrelated document bodies.
        found = next(((path, relative) for path, relative in self._paths(source)
                      if document_id is None
                      or _document_id(self.vault_id, relative) == document_id), None)
        if found is None:
            raise DocumentNotFound(f'No document matches document_id={document_id!r}, source={source!r}.')
        path, relative = found
        try:
            note = _load_note(path, source=relative)
        except FileNotFoundError as error:
            raise DocumentNotFound(f'Document no longer exists: {relative}.') from error
        heading_path = ()
        if section_id is not None:
            section = next((s for s in _sections(note) if s.section_id == section_id), None)
            if section is None:
                raise DocumentNotFound(f'Unknown section: {section_id}.')
            start, end = section.blocks[0].start, section.blocks[-1].end
            heading_path = section.heading_path
        else:
            start = 0 if start_char is None else start_char
            end = len(note.content) if end_char is None else end_char
            if not 0 <= start <= end <= len(note.content):
                raise ValueError('Character range is outside the current document body.')
        return DocumentSlice(_document_id(self.vault_id, note.source), note.document_revision,
                             note.source, note.title, note.content[start:end], start, end,
                             section_id, heading_path)


def scan_notes(directory: Path, *, exclude: Sequence[str] = DEFAULT_EXCLUDES) -> VaultScan:
    """Read the Markdown scope for indexing; fail if its inventory changes during scanning.

    One unusable file must not cost a whole vault its index: a file that cannot
    be read or decoded is skipped and reported, and a file whose frontmatter
    falls outside the parsed YAML subset is loaded without metadata and
    reported. A file appearing, disappearing or changing mid-scan still fails
    the scan, because the caller is about to build an index from it.
    """
    def inventory():
        result = {}
        for path, source in note_files(directory, exclude=exclude):
            stat = path.stat()
            result[source] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return result
    before = inventory()
    notes, unreadable, unparsed = [], [], []
    for path, source in note_files(directory, exclude=exclude):
        try:
            note, error = _read_note(path, source)
        except (OSError, UnicodeDecodeError) as failure:
            unreadable.append(SkippedNote(source, f'{type(failure).__name__}: {failure}'))
            continue
        notes.append(note)
        if error is not None:
            unparsed.append(SkippedNote(source, error))
    seen = {note.source for note in notes} | {skipped.source for skipped in unreadable}
    if before != inventory() or seen != set(before):
        raise ValueError('Notes changed during scanning; rerun the index command.')
    return VaultScan(tuple(notes), tuple(unreadable), tuple(unparsed))

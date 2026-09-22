"""Resolve the note-to-note link graph of a Markdown vault.

Obsidian's strongest structural signal is the link, and it is written in two
forms. Both are recognized here with regular expressions rather than a Markdown
parser, because only the link target matters and the vault decides what a
target means.

Handled:

* ``[[note]]``, ``[[folder/note]]``, ``[[note.md]]``
* ``[[note|alias]]`` -- the alias is display text; the left side is the target
* ``[[note#heading]]`` and ``[[note#^block-id]]`` -- the fragment is dropped
* ``![[note]]`` -- an embed of another note is a link to it
* ``[text](folder/note.md)``, ``[text](<a note.md>)``, ``[text](note.md#heading)``
  and percent-encoded spellings of the same

Ignored, deliberately:

* any target that does not resolve to a note in this scan: a link to a note
  that does not exist yet, an attachment, an external URL, a ``mailto:`` or
  ``obsidian://`` target, a bare ``#heading`` link inside the same note, and a
  relative path that climbs out of the vault
* a Markdown link whose target does not end in ``.md``, and a reference-style
  link (``[text][label]``), whose target lives in a separate definition
* links written inside fenced code blocks: a fence is not tracked, but a
  literal such as ``[[ -f file ]]`` resolves to no note and drops out anyway
* links in the YAML frontmatter, which is removed from the body before parsing
* a note's link to itself

Names are resolved the way the vault spells them, not the way a filesystem
does: see :func:`_Vault.resolve` for the order and for how a name shared by
several notes is settled.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
import posixpath
import re
from urllib.parse import unquote

from arkb.knowledge.models import Note


# Identifies what the stored graph means. An index built by an earlier parser
# records a different value (or none), which denies snapshot reuse and rebuilds
# the graph instead of serving one produced by rules that no longer hold.
LINK_PARSER_VERSION = 'wikilink-v1'

CONTEXT_CHARS = 180

_WIKILINK = re.compile(r'!?\[\[([^\[\]\n]+?)\]\]')
_MARKDOWN_LINK = re.compile(
    r'!?\[[^\]\n]*\]\(\s*(?:<([^<>\n]*)>|([^\s()]*))(?:\s+"[^"\n]*")?\s*\)')
_SCHEME = re.compile(r'[A-Za-z][A-Za-z0-9+.\-]*:')


@dataclass(frozen=True)
class OutgoingLink:
    """One resolved edge: the linking note, the note it points at, its context.

    ordinal orders a note's outgoing links by first appearance in its body.
    occurrences counts how often the same note is linked from it; context is a
    short excerpt around the first occurrence, enough to see why the link is
    there without becoming quotable evidence.
    """

    source: str
    target: str
    ordinal: int
    occurrences: int
    context: str


@dataclass(frozen=True)
class NoteLink:
    """One end of a link as a caller sees it: the note at the other side."""

    source: str
    occurrences: int
    context: str


def _link_targets(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield each raw link target with the span it was written in."""
    for match in _WIKILINK.finditer(text):
        target = match.group(1).split('|', 1)[0].split('#', 1)[0].strip()
        if target:
            yield target, match.start(), match.end()
    for match in _MARKDOWN_LINK.finditer(text):
        raw = match.group(1) if match.group(1) is not None else match.group(2)
        target = unquote(raw or '').split('#', 1)[0].strip()
        if target and not target.startswith('//') and _SCHEME.match(target) is None:
            yield target, match.start(), match.end()


def _context(text: str, start: int, end: int, *, width: int = CONTEXT_CHARS) -> str:
    """Excerpt the line holding a link, narrowed to a window around long lines."""
    line_start = text.rfind('\n', 0, start) + 1
    line_end = text.find('\n', end)
    line_end = len(text) if line_end < 0 else line_end
    if line_end - line_start <= width:
        return ' '.join(text[line_start:line_end].split())
    margin = max(0, (width - (end - start)) // 2)
    left, right = max(line_start, start - margin), min(line_end, end + margin)
    excerpt = ' '.join(text[left:right].split())
    return ('...' if left > line_start else '') + excerpt + ('...' if right < line_end else '')


def _alias_values(note: Note) -> Iterator[str]:
    for key in ('aliases', 'alias'):
        value = note.metadata.get(key)
        for item in value if isinstance(value, list) else [value] if isinstance(value, str) else []:
            if isinstance(item, str) and item.strip():
                yield item.strip()


class _Vault:
    """Every spelling that can address a note in one scan, built once per index."""

    def __init__(self, notes: Sequence[Note]):
        self.sources = {note.source for note in notes}
        self.by_path: dict[str, list[str]] = {}
        self.by_name: dict[str, list[str]] = {}
        self.by_alias: dict[str, list[str]] = {}
        for note in notes:
            self.by_path.setdefault(note.source.lower(), []).append(note.source)
            stem = PurePosixPath(note.source).stem
            self.by_name.setdefault(stem.lower(), []).append(note.source)
            for alias in _alias_values(note):
                self.by_alias.setdefault(alias.lower(), []).append(note.source)

    @staticmethod
    def _closest(candidates: Sequence[str], origin: str) -> str:
        """Settle a name shared by several notes the way a reader would read it.

        Obsidian lets a link name a note without its folder, so one name can
        address several notes. Prefer one in the linking note's own folder,
        then the shallowest path, then the alphabetically first source. The
        choice is deterministic, so the same vault always yields the same
        graph; it can still differ from what Obsidian's own resolver shows for
        a genuinely ambiguous name.
        """
        origin_directory = posixpath.dirname(origin)
        return min(candidates, key=lambda source: (posixpath.dirname(source) != origin_directory,
                                                   source.count('/'), source))

    def _spellings(self, target: str, origin: str) -> Iterator[str]:
        forms = [target] if target.lower().endswith('.md') else [target, target + '.md']
        directory = posixpath.dirname(origin)
        for form in forms:
            for candidate in (form, posixpath.join(directory, form) if directory else form):
                normalized = posixpath.normpath(candidate)
                if not normalized.startswith(('/', '../')) and normalized not in ('.', '..'):
                    yield normalized

    def resolve(self, target: str, *, origin: str) -> str | None:
        """Resolve one written target to a source in this scan, or to nothing.

        In order: the target read as a vault-relative path, then as a path
        relative to the linking note, in both cases with ``.md`` supplied when
        it is missing and ignoring case; then the target read as a bare note
        name; then as a note's frontmatter alias. A target that matches none of
        these -- an unresolved link, an attachment, anything outside the vault
        -- returns None and is left out of the graph.
        """
        if '\\' in target or target.startswith('#'):
            return None
        for spelling in self._spellings(target, origin):
            if spelling in self.sources:
                return spelling
            candidates = self.by_path.get(spelling.lower())
            if candidates:
                return self._closest(candidates, origin)
        name = PurePosixPath(target).name
        name = name[:-3] if name.lower().endswith('.md') else name
        for index in (self.by_name, self.by_alias):
            candidates = index.get(name.lower())
            if candidates:
                return self._closest(candidates, origin)
        return None


def resolve_links(notes: Sequence[Note]) -> tuple[OutgoingLink, ...]:
    """Build the outgoing link graph of one complete scan, in source order.

    Backlinks are not returned: they are this relation read backwards, so
    storing them again would only create a second thing to keep consistent.
    """
    vault = _Vault(notes)
    graph: list[OutgoingLink] = []
    for note in notes:
        found: dict[str, OutgoingLink] = {}
        for target, start, end in sorted(_link_targets(note.content), key=lambda item: item[1]):
            resolved = vault.resolve(target, origin=note.source)
            if resolved is None or resolved == note.source:
                continue
            if resolved in found:
                found[resolved] = replace(found[resolved], occurrences=found[resolved].occurrences + 1)
                continue
            found[resolved] = OutgoingLink(source=note.source, target=resolved, ordinal=len(found),
                                           occurrences=1, context=_context(note.content, start, end))
        graph.extend(found.values())
    return tuple(graph)


class LinkGraph:
    """The link graph of one published snapshot, queried a note at a time.

    The graph describes the vault as it was indexed, while read and list work
    on live files: a note linked after the last index is found by search, not
    here. Backlinks are the stored relation read backwards.
    """

    def __init__(self, storage, index_version: str):
        self.storage = storage
        self.index_version = index_version

    def out(self, source: str) -> list[NoteLink]:
        return [NoteLink(link.target, link.occurrences, link.context)
                for link in self.storage.note_links(self.index_version, source=source)]

    def incoming(self, source: str) -> list[NoteLink]:
        return [NoteLink(link.source, link.occurrences, link.context)
                for link in self.storage.note_links(self.index_version, target=source)]

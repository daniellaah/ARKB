from pathlib import Path

import pytest

from arkb.knowledge.documents import DocumentAccess, load_notes


def test_load_notes_reads_title_content_and_source(tmp_path: Path) -> None:
    (tmp_path / "reading.md").write_text(
        "# Reading Notes\n\n"
        "Keep the author's meaning.\n\n"
        "## Source\n\n"
        "Sönke Ahrens.\n",
        encoding="utf-8",
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content, note.source) for note in notes] == [
        (
            "Reading Notes",
            "Keep the author's meaning.\n\n## Source\n\nSönke Ahrens.",
            "reading.md",
        )
    ]


def test_load_notes_returns_notes_in_filename_order(tmp_path: Path) -> None:
    for filename in ["zeta.md", "alpha.md", "Beta.md"]:
        (tmp_path / filename).write_text("# A Note\n\nAn idea.\n", encoding="utf-8")

    notes = load_notes(tmp_path)

    assert [note.source for note in notes] == ["Beta.md", "alpha.md", "zeta.md"]


def test_load_notes_uses_filename_when_no_level_one_heading_exists(
    tmp_path: Path,
) -> None:
    (tmp_path / "reading_notes.md").write_text(
        "## Reading\n\nA useful passage.\n", encoding="utf-8"
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content) for note in notes] == [
        ("reading_notes", "## Reading\n\nA useful passage.")
    ]


def test_load_notes_uses_first_level_one_heading_and_preserves_other_content(
    tmp_path: Path,
) -> None:
    (tmp_path / "ideas.md").write_text(
        "Introductory text.\n"
        "# Main Idea\n\n"
        "Develop one idea.\n\n"
        "# Another Heading\n\n"
        "Keep this section.\n",
        encoding="utf-8",
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content) for note in notes] == [
        (
            "Main Idea",
            "Introductory text.\n\nDevelop one idea.\n\n"
            "# Another Heading\n\nKeep this section.",
        )
    ]


def test_load_notes_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_notes(tmp_path / "missing")


def test_load_notes_reads_only_markdown_files_including_nested_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "current.md").write_text("# Current\n\nAn idea.\n", encoding="utf-8")
    (tmp_path / "draft.txt").write_text("An unfinished draft.\n", encoding="utf-8")
    archive = tmp_path / "archive.md"
    archive.mkdir()
    (archive / "old.md").write_text("# Old\n\nAn earlier idea.\n", encoding="utf-8")

    notes = load_notes(tmp_path)

    assert [note.source for note in notes] == ["archive.md/old.md", "current.md"]


def test_load_notes_returns_an_empty_list_for_an_empty_directory(tmp_path: Path) -> None:
    assert load_notes(tmp_path) == []


def test_scan_rejects_changes_during_reading(tmp_path, monkeypatch):
    import arkb.knowledge.documents as loaders
    (tmp_path / 'a.md').write_text('# A\nbody')
    original = loaders._read_note
    def changing(path, source):
        (path.parent / 'b.md').write_text('# B\nnew')
        return original(path, source)
    monkeypatch.setattr(loaders, '_read_note', changing)
    with pytest.raises(ValueError, match='changed during scanning'):
        loaders.scan_notes(tmp_path)


def test_access_identity_matches_index_records_and_only_reads_resolved_file(tmp_path, monkeypatch):
    from unittest.mock import Mock

    import arkb.knowledge.documents as documents
    from arkb.knowledge.chunking import whole_note_chunks
    from arkb.knowledge.models import ChunkRecord

    (tmp_path / 'a.md').write_text('# A\n\nbody', encoding='utf-8')
    note, = load_notes(tmp_path)
    indexed = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id='v')
    (tmp_path / 'unrelated.md').write_bytes(b'\xff')
    load = Mock(wraps=documents._load_note)
    monkeypatch.setattr(documents, '_load_note', load)
    access = DocumentAccess(tmp_path, vault_id='v')
    read = access.read(indexed.document_id)
    assert (read.document_id, read.document_revision, read.source, read.title, read.content) == (
        indexed.document_id, indexed.document_revision, note.source, note.title, note.content)
    assert not hasattr(read, 'chunk_id')
    load.assert_called_once_with(tmp_path / 'a.md', source='a.md')
    load.reset_mock()
    assert access.read(source='a.md') == read
    load.assert_called_once_with(tmp_path / 'a.md', source='a.md')
    load.reset_mock()
    with pytest.raises(LookupError):
        access.read(indexed.document_id, source='unrelated.md')
    load.assert_not_called()
    with pytest.raises(LookupError):
        DocumentAccess(tmp_path, vault_id='other').read(indexed.document_id)


def test_access_sections_reuse_chunker_coordinates_and_ids(tmp_path):
    from arkb.knowledge.chunking import chunk_notes

    text = '# Title\n\nintro\n## One\nbody\n### Child\nchild body\n## One\nsecond body'
    (tmp_path / 'a.md').write_text(text, encoding='utf-8')
    access = DocumentAccess(tmp_path, vault_id='v')
    whole = next(access.records())
    chunks = chunk_notes(load_notes(tmp_path), count_tokens=len, chunk_size=100, chunk_overlap=0)
    for chunk in chunks:
        read = access.read(whole.document_id, section_id=chunk.section_id)
        assert read.content == chunk.content
        assert read.heading_path == chunk.heading_path
        assert read.section_id == chunk.section_id
        assert (read.start_char, read.end_char) == (chunk.start_char, chunk.end_char)
    assert access.read(whole.document_id).content == whole.chunk.content


def test_access_tracks_new_deleted_renamed_files_without_retaining_bodies(tmp_path):
    access = DocumentAccess(tmp_path, vault_id='v')
    assert list(access.records()) == []
    path = tmp_path / 'a.md'
    path.write_text('# A\nold', encoding='utf-8')
    original = next(access.records())
    path.write_text('# A\nnew', encoding='utf-8')
    current = access.read(original.document_id)
    assert current.content == 'new'
    assert current.document_revision != original.document_revision
    path.rename(tmp_path / 'renamed.md')
    with pytest.raises(LookupError):
        access.read(original.document_id)
    assert next(access.records()).document_id != original.document_id


def test_access_spans_subdirectories_and_excludes_external_symlinks(tmp_path):
    from arkb.knowledge.documents import scan_notes
    root = tmp_path / 'notes'
    root.mkdir()
    (root / 'a.md').write_text('inside', encoding='utf-8')
    (root / 'ignore.txt').write_text('ignore', encoding='utf-8')
    (root / 'nested').mkdir()
    (root / 'nested' / 'nested.md').write_text('nested', encoding='utf-8')
    outside = tmp_path / 'private.md'
    outside.write_text('outside', encoding='utf-8')
    (root / 'link.md').symlink_to(outside)
    # Indexing retains its existing symlink scope; live tools stay confined.
    assert [note.source for note in load_notes(root)] == ['a.md', 'link.md', 'nested/nested.md']
    assert list(scan_notes(root).notes) == load_notes(root)
    access = DocumentAccess(root, vault_id='v')
    assert [r.chunk.source for r in access.records()] == ['a.md', 'nested/nested.md']
    assert list(access.records(source='../private.md')) == []
    assert access.read(source='a.md').content == 'inside'
    assert access.read(source='nested/nested.md').content == 'nested'
    for source in ('../private.md', 'nested/../nested/nested.md', str(outside), 'link.md',
                   'nested\\nested.md', 'ignore.txt'):
        with pytest.raises(LookupError):
            access.read(source=source)


def test_access_propagates_filesystem_and_decoding_errors(tmp_path, monkeypatch):
    access = DocumentAccess(tmp_path, vault_id='v')
    (tmp_path / 'a.md').write_text('body', encoding='utf-8')
    record = next(access.records())
    (tmp_path / 'a.md').write_bytes(b'\xff')
    with pytest.raises(UnicodeDecodeError):
        access.read(record.document_id)
    with pytest.raises(FileNotFoundError):
        list(DocumentAccess(tmp_path / 'missing', vault_id='v').records())

    def disappeared(path, *, source=None):
        raise FileNotFoundError('removed during read')
    monkeypatch.setattr('arkb.knowledge.documents._load_note', disappeared)
    with pytest.raises(LookupError, match='no longer exists'):
        access.read(record.document_id)


def test_note_files_walks_subdirectories_in_source_order_and_skips_excluded_folders(tmp_path):
    from arkb.knowledge.documents import DEFAULT_EXCLUDES, note_files
    for relative in ('top.md', '04-Areas/Career Development/note.md', '04-Areas/other.md',
                     '.obsidian/plugin.md', '.trash/deleted.md', 'Attachments/clip.md',
                     'Excalidraw/sketch.md', 'Archive/old.md', 'Archive/nested/older.md'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# {relative}\n\nbody', encoding='utf-8')
    (tmp_path / '04-Areas/image.png').write_bytes(b'\x89PNG')

    assert [source for _, source in note_files(tmp_path)] == [
        '04-Areas/Career Development/note.md', '04-Areas/other.md',
        'Archive/nested/older.md', 'Archive/old.md', 'top.md']
    assert [source for _, source in note_files(tmp_path, exclude=(*DEFAULT_EXCLUDES, 'Archive'))] == [
        '04-Areas/Career Development/note.md', '04-Areas/other.md', 'top.md']
    # A pattern may address one nested folder without excluding every folder of that name.
    assert [source for _, source in note_files(tmp_path, exclude=(*DEFAULT_EXCLUDES, 'Archive/nested'))] == [
        '04-Areas/Career Development/note.md', '04-Areas/other.md', 'Archive/old.md', 'top.md']
    assert [source for _, source in note_files(tmp_path, exclude=())] == [
        '.obsidian/plugin.md', '.trash/deleted.md', '04-Areas/Career Development/note.md',
        '04-Areas/other.md', 'Archive/nested/older.md', 'Archive/old.md',
        'Attachments/clip.md', 'Excalidraw/sketch.md', 'top.md']


def test_excluded_directories_leave_the_live_document_scope(tmp_path):
    (tmp_path / 'Attachments').mkdir()
    (tmp_path / 'Attachments' / 'clip.md').write_text('# Clip\n\nbody', encoding='utf-8')
    (tmp_path / 'kept.md').write_text('# Kept\n\nbody', encoding='utf-8')
    access = DocumentAccess(tmp_path, vault_id='v')
    assert [note['source'] for note in access.list()['notes']] == ['kept.md']
    assert list(access.records(source='Attachments/clip.md')) == []
    with pytest.raises(LookupError):
        access.read(source='Attachments/clip.md')
    included = DocumentAccess(tmp_path, vault_id='v', exclude=())
    assert included.read(source='Attachments/clip.md').content == 'body'


def test_identical_filenames_in_different_folders_are_distinct_documents(tmp_path):
    for folder in ('one', 'two'):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'note.md').write_text(f'# Note\n\n{folder} body', encoding='utf-8')
    access = DocumentAccess(tmp_path, vault_id='v')
    records = list(access.records())
    assert [record.chunk.source for record in records] == ['one/note.md', 'two/note.md']
    assert records[0].document_id != records[1].document_id
    assert access.read(records[0].document_id).content == 'one body'
    assert access.read(records[1].document_id).content == 'two body'
    assert access.read(source='two/note.md').document_id == records[1].document_id


def test_load_note_strips_frontmatter_into_metadata_without_changing_title_rules(tmp_path):
    (tmp_path / 'a.md').write_text(
        '---\n'
        'title: Ignored\n'
        'tags:\n'
        '  - career\n'
        '  - "writing"\n'
        'aliases: [Plan B, second]\n'
        'created: 2026-09-22\n'
        'empty:\n'
        '# a comment\n'
        '---\n'
        '# Real Title\n\nBody text.\n', encoding='utf-8')
    (tmp_path / 'b.md').write_text('---\ntags: one\n---\nNo heading here.\n', encoding='utf-8')
    note, other = load_notes(tmp_path)
    assert note.metadata == {'title': 'Ignored', 'tags': ['career', 'writing'],
                             'aliases': ['Plan B', 'second'], 'created': '2026-09-22', 'empty': ''}
    assert (note.title, note.content) == ('Real Title', 'Body text.')
    assert (other.title, other.content, other.metadata) == ('b', 'No heading here.', {'tags': 'one'})


def test_frontmatter_is_only_stripped_when_the_block_is_delimited(tmp_path):
    (tmp_path / 'rule.md').write_text('---\nA horizontal rule opens this note.\n', encoding='utf-8')
    (tmp_path / 'plain.md').write_text('# Plain\n\nNo block.\n', encoding='utf-8')
    plain, rule = load_notes(tmp_path)
    assert rule.content == '---\nA horizontal rule opens this note.' and rule.metadata == {}
    assert plain.content == 'No block.' and plain.metadata == {}


def test_unparsable_frontmatter_keeps_the_body_and_is_reported(tmp_path):
    from arkb.knowledge.documents import scan_notes
    (tmp_path / 'bad.md').write_text(
        '---\nnested:\n  key: value\n---\n# Bad\n\nStill indexed.\n', encoding='utf-8')
    (tmp_path / 'good.md').write_text('---\ntags: t\n---\n# Good\n\nFine.\n', encoding='utf-8')
    scan = scan_notes(tmp_path)
    assert [(note.source, note.title, note.content) for note in scan.notes] == [
        ('bad.md', 'Bad', 'Still indexed.'), ('good.md', 'Good', 'Fine.')]
    assert scan.notes[0].metadata == {} and scan.notes[1].metadata == {'tags': 't'}
    assert [skipped.source for skipped in scan.unparsed_metadata] == ['bad.md']
    assert 'Nested mappings' in scan.unparsed_metadata[0].reason
    assert scan.unreadable == ()


def test_scan_skips_and_reports_one_unreadable_file_instead_of_failing(tmp_path):
    from arkb.knowledge.documents import scan_notes
    (tmp_path / 'good.md').write_text('# Good\n\nbody', encoding='utf-8')
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'broken.md').write_bytes(b'# Broken\n\n\xff\xfe')
    scan = scan_notes(tmp_path)
    assert [note.source for note in scan.notes] == ['good.md']
    assert [skipped.source for skipped in scan.unreadable] == ['sub/broken.md']
    assert 'UnicodeDecodeError' in scan.unreadable[0].reason
    with pytest.raises(UnicodeDecodeError):
        load_notes(tmp_path)


@pytest.mark.parametrize('source', [
    '../outside.md', '/abs/note.md', 'C:/vault/note.md', 'sub\\note.md',
    './sub/note.md', 'sub//note.md', 'sub/../sub/note.md', 'sub/note.txt'])
def test_access_rejects_non_canonical_source_selectors(tmp_path, source):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'note.md').write_text('# Note\n\nbody', encoding='utf-8')
    access = DocumentAccess(tmp_path, vault_id='v')
    assert list(access.records(source=source)) == []
    with pytest.raises(LookupError):
        access.read(source=source)


def test_list_filters_by_path_tag_and_modification_time(linked_vault):
    access = DocumentAccess(linked_vault, vault_id='v')

    def sources(**filters):
        return [note['source'] for note in access.list(**filters)['notes']]

    # A pattern now filters the vault-relative path, so a folder is selectable.
    assert sources(pattern='Concepts/*') == ['Concepts/Attention.md', 'Concepts/KV Cache.md']
    assert sources(pattern='kv') == ['Archive/KV Cache.md', 'Concepts/KV Cache.md']
    # A tag matches frontmatter or body, ignoring case and a leading "#", and a
    # parent tag matches the nested tags below it.
    assert sources(tag='#MOC') == ['index.md']
    assert sources(tag='ai') == ['Concepts/Attention.md', 'Concepts/KV Cache.md']
    assert sources(tag='ai/llm') == ['Concepts/KV Cache.md']
    assert sources(tag='daily') == ['Journal/2026-01-02.md']
    assert sources(tag='archive') == ['Archive/Attention.md', 'Archive/KV Cache.md']
    # Dates bound the file's modification time: at or after, strictly before.
    assert sources(modified_after='2026-03-01') == ['Concepts/KV Cache.md', 'index.md']
    assert sources(modified_before='2026-01-05') == ['Journal/2026-01-02.md']
    assert sources(modified_after='2026-01-05', modified_before='2026-02-11') == [
        'Archive/Attention.md', 'Archive/KV Cache.md', 'Concepts/Attention.md']
    # Filters combine with AND.
    assert sources(pattern='Concepts/*', tag='ai/transformer') == ['Concepts/Attention.md']
    assert sources(pattern='Archive/*', modified_after='2026-02-01') == []


def test_list_reports_each_note_with_its_tags_and_modification_time(linked_vault):
    listing = DocumentAccess(linked_vault, vault_id='v').list(tag='ai', limit=1)

    note, = listing['notes']
    assert note['source'] == 'Concepts/Attention.md' and note['tags'] == ['ai/transformer']
    assert note['modified'] == '2026-02-10T12:00:00' and note['headings'] == ['## Masking']
    assert listing['total'] == 2 and listing['truncated'] is True
    # A note without tags does not carry an empty list.
    plain = DocumentAccess(linked_vault, vault_id='v').list(pattern='Journal/*')['notes'][0]
    assert plain['tags'] == ['journal', 'daily'] and 'more_headings' not in plain


def test_list_rejects_a_date_that_is_not_ISO(linked_vault):
    access = DocumentAccess(linked_vault, vault_id='v')
    for filters in ({'modified_after': 'yesterday'}, {'modified_before': '2026-13-01'}):
        with pytest.raises(ValueError, match='ISO date'):
            access.list(**filters)
    with pytest.raises(ValueError):
        access.list(tag=' ')


def test_titles_name_known_sources_and_skip_the_rest(linked_vault):
    access = DocumentAccess(linked_vault, vault_id='v')

    assert access.titles(['index.md', 'Concepts/Attention.md', 'gone.md', '../outside.md']) == {
        'index.md': 'Vault Map', 'Concepts/Attention.md': 'Attention'}

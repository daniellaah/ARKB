"""Resolving the vault's link graph, and reading it back out of a snapshot."""

import pytest

from arkb.knowledge.documents import load_notes
from arkb.knowledge.links import LinkGraph, resolve_links
from arkb.knowledge.models import EmbeddingSpec, IndexManifest, Note, fingerprint_config
from arkb.knowledge.sqlite import SQLiteStorage


def edges(notes):
    return [(link.source, link.target) for link in resolve_links(notes)]


def test_every_written_link_form_resolves_and_the_rest_is_left_out(linked_vault):
    notes = load_notes(linked_vault)

    assert edges(notes) == [
        # A bare name in the linking note's own folder wins over the same name elsewhere.
        ('Archive/Attention.md', 'Archive/KV Cache.md'),
        ('Archive/KV Cache.md', 'index.md'),
        # [[Attention#Masking]] drops its heading; the second mention folds into the same edge.
        ('Concepts/KV Cache.md', 'Concepts/Attention.md'),
        # A Markdown link relative to the linking note, with a fragment.
        ('Concepts/KV Cache.md', 'Journal/2026-01-02.md'),
        # [[KV|key-value cache]]: the display alias is dropped and "KV" is a frontmatter alias.
        ('Journal/2026-01-02.md', 'Concepts/KV Cache.md'),
        # Two notes are called Attention; neither is in Journal/, so the shorter
        # path wins and ties are settled alphabetically.
        ('Journal/2026-01-02.md', 'Archive/Attention.md'),
        ('index.md', 'Concepts/KV Cache.md'),
        # [[2026-01-02|latest entry]] names a note in another folder.
        ('index.md', 'Journal/2026-01-02.md'),
    ]
    # Absent by design: [[Missing Note]] (no such note), ![[diagram.png]] (not a
    # note), an https:// paper, [[index]] inside index.md, and both Markdown
    # links that climb out of the vault.
    targets = {target for _, target in edges(notes)}
    assert 'Concepts/Attention.md' in targets and not any(t.endswith('other.md') for t in targets)


def test_repeated_links_keep_document_order_and_count_their_occurrences(linked_vault):
    links = {(link.source, link.target): link for link in resolve_links(load_notes(linked_vault))}

    attention = links[('Concepts/KV Cache.md', 'Concepts/Attention.md')]
    journal = links[('Concepts/KV Cache.md', 'Journal/2026-01-02.md')]
    assert (attention.ordinal, attention.occurrences) == (0, 2)
    assert (journal.ordinal, journal.occurrences) == (1, 1)
    # The context is the line the first occurrence was written in, not evidence.
    assert attention.context.startswith('Keys and values are kept') and '[[Attention#Masking]]' in attention.context


def test_a_long_line_is_narrowed_around_the_link():
    filler = 'padding words ' * 40
    note = Note(title='T', content=f'{filler}[[Other]]{filler}', source='a.md')
    link, = resolve_links([note, Note(title='O', content='body', source='Other.md')])

    assert link.context.startswith('...') and link.context.endswith('...')
    assert '[[Other]]' in link.context and len(link.context) < len(note.content)


@pytest.mark.parametrize('target', [
    'https://example.com/page.md',            # an external address
    'mailto:someone@example.com',             # another scheme
    '../../outside/note.md',                  # above the vault root
    'Attachments/diagram.png',                # not a note
    'Missing Note',                           # not written yet
])
def test_targets_outside_the_scan_are_ignored(target):
    notes = [Note(title='A', content=f'See [link]({target}) and [[{target}]].', source='a.md'),
             Note(title='B', content='body', source='folder/b.md')]

    assert resolve_links(notes) == ()


def test_frontmatter_links_and_self_links_stay_out(tmp_path):
    (tmp_path / 'a.md').write_text('---\nrelated: "[[b]]"\n---\n# A\nSee [[a]] and [[b]].\n', encoding='utf-8')
    (tmp_path / 'b.md').write_text('# B\n', encoding='utf-8')

    # The body link to b resolves once; the frontmatter mention is not parsed
    # and the note's link to itself carries nowhere.
    assert edges(load_notes(tmp_path)) == [('a.md', 'b.md')]
    assert resolve_links(load_notes(tmp_path))[0].occurrences == 1


def graph(path, notes):
    """Store one scan's links in a building snapshot and read them back."""
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1')
    manifest = IndexManifest(index_version='v1', vault_id='vault', embedding_spec=spec,
                             chunking_fingerprint=fingerprint_config({}),
                             document_count=0, chunk_count=0)
    storage = SQLiteStorage(path)
    storage.create_build(manifest, corpus_fingerprint='corpus', backend={'kind': 'qdrant'})
    storage.add_links('v1', resolve_links(notes))
    return storage, LinkGraph(storage, 'v1')


def test_a_snapshot_serves_outgoing_links_and_backlinks_read_in_reverse(tmp_path, linked_vault):
    storage, links = graph(tmp_path / 'index.sqlite', load_notes(linked_vault))
    with storage:
        outgoing = links.out('Concepts/KV Cache.md')
        incoming = links.incoming('Concepts/KV Cache.md')

        assert [link.source for link in outgoing] == ['Concepts/Attention.md', 'Journal/2026-01-02.md']
        assert outgoing[0].occurrences == 2
        # Backlinks are the same rows read by target, ordered by the linking note.
        assert [link.source for link in incoming] == ['Journal/2026-01-02.md', 'index.md']
        assert 'key-value cache' in incoming[0].context
        assert links.out('Concepts/Attention.md') == []
        assert [link.source for link in links.incoming('Concepts/Attention.md')] == ['Concepts/KV Cache.md']


def test_link_rows_belong_to_their_build_and_require_a_query_selector(tmp_path, linked_vault):
    notes = load_notes(linked_vault)
    storage, _ = graph(tmp_path / 'index.sqlite', notes)
    with storage:
        with pytest.raises(ValueError, match='by source or by target'):
            storage.note_links('v1')
        with pytest.raises(ValueError, match='by source or by target'):
            storage.note_links('v1', source='index.md', target='index.md')
        # A published snapshot is immutable: links belong to a build in progress.
        storage.publish('v1')
        with pytest.raises(ValueError, match='Only building snapshots'):
            storage.add_links('v1', resolve_links(notes))
        assert storage.note_links('v1', target='index.md')[0].source == 'Archive/KV Cache.md'


def test_a_repeated_pair_is_stored_once(tmp_path):
    notes = [Note(title='A', content='[[b]] then [[b]] again', source='a.md'),
             Note(title='B', content='body', source='b.md')]
    storage, links = graph(tmp_path / 'index.sqlite', notes)
    with storage:
        stored, = storage.note_links('v1', source='a.md')
        assert (stored.target, stored.occurrences) == ('b.md', 2)
        assert [(link.source, link.occurrences) for link in links.out('a.md')] == [('b.md', 2)]

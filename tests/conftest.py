from contextlib import closing
from datetime import datetime
import os
from textwrap import dedent
import warnings

import pytest
from qdrant_client import QdrantClient


@pytest.fixture
def qdrant():
    """Exercise the Qdrant API locally; real ANN checks remain server tests."""
    with warnings.catch_warnings(), closing(QdrantClient(':memory:')) as client:
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        warnings.filterwarnings('ignore', message='Local mode performs exact.*')
        yield client


@pytest.fixture
def qdrant_config():
    from arkb.knowledge.models import QdrantConfig
    return QdrantConfig()


# One small Obsidian-shaped vault: folders, frontmatter, tags, aliases, every
# link form worth resolving and several worth ignoring. The evaluation corpus
# is a flat regression baseline with no links, so link, tag and date behaviour
# is covered here instead. Modification times are fixed so date filters can be
# asserted exactly.
LINKED_NOTES = {
    'index.md': ("""\
        ---
        tags: [map, moc]
        ---
        # Vault Map
        The map starts at [[Concepts/KV Cache]] and the [[2026-01-02|latest entry]].
        It links to [[index]] itself and to [an outside note](../outside/other.md).
        """, datetime(2026, 4, 1, 9, 30)),
    'Concepts/KV Cache.md': ("""\
        ---
        tags:
          - ai/llm
        aliases: [KV, key-value cache]
        ---
        # KV Cache
        Keys and values are kept between steps; see [[Attention#Masking]] for why that is safe.
        A second mention of [[Attention]] belongs to the same pair of notes.
        An attachment ![[diagram.png]] and a paper [Vaswani](https://arxiv.org/abs/1706.03762).
        Measured on the [2026-01-02](../Journal/2026-01-02.md#morning) run.
        """, datetime(2026, 3, 15, 8, 0)),
    'Concepts/Attention.md': ("""\
        # Attention
        Scaled dot-product attention, tagged #ai/transformer in the body.
        ## Masking
        Causal masks hide the future, and [[Missing Note]] is not written yet.
        """, datetime(2026, 2, 10, 12, 0)),
    'Archive/Attention.md': ("""\
        # Attention (archived)
        An older draft, tagged #archive, that still points at [[KV Cache]].
        """, datetime(2026, 1, 5, 7, 0)),
    'Archive/KV Cache.md': ("""\
        # KV Cache (archived)
        Superseded, tagged #archive; the current map is [[index]].
        """, datetime(2026, 1, 5, 7, 0)),
    'Journal/2026-01-02.md': ("""\
        ---
        tags: journal, daily
        ---
        # 2026-01-02
        Read about the [[KV|key-value cache]] and revisited [[Attention]].
        Outside the vault: [elsewhere](../../elsewhere/other.md).
        """, datetime(2026, 1, 2, 21, 15)),
}


@pytest.fixture
def linked_vault(tmp_path):
    vault = tmp_path / 'vault'
    for source, (body, modified) in LINKED_NOTES.items():
        path = vault / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dedent(body), encoding='utf-8')
        os.utime(path, (modified.timestamp(), modified.timestamp()))
    return vault

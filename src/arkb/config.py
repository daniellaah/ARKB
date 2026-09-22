"""Explicit application settings; importing configuration opens no resources."""

from dataclasses import dataclass
import os
from pathlib import Path

DEFAULT_EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
DEFAULT_GENERATION_MODEL = 'qwen3.5:4b'
DEFAULT_AGENT_THINK = True
DEFAULT_RETRIEVAL_MODE = 'semantic'
DEFAULT_DB = Path('.arkb/index.sqlite')
DEFAULT_NOTES_DIR = Path('example_notes')


@dataclass(frozen=True, kw_only=True)
class RuntimeConfig:
    host: str = 'http://127.0.0.1:11434'
    timeout: float = 180.0
    tokenizer_cache: Path | None = None
    offline: bool = False
    embedding_model: str | None = None
    qdrant_url: str | None = None
    qdrant_timeout: float | None = None


@dataclass(frozen=True, kw_only=True)
class RetrievalConfig:
    candidate_k: int = 20
    rrf_k: float = 60
    rerank_candidates: int = 20
    reranker_max_length: int = 512
    reranker_cache: str | None = None
    bm25_k1: float = 1.2
    bm25_b: float = .75


def load_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines into the process environment without overriding variables already set.

    Blank lines and # comments are skipped; an optional `export ` prefix and
    surrounding single or double quotes are removed. Returns the variables set.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    loaded = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.removeprefix('export ').strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded

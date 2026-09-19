"""Shared paths and helpers for the agentic evaluation library and its studies."""
from datetime import datetime, timezone
import json
from pathlib import Path

from arkb.evaluation.external import write_json

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'evaluation/agentic-tools/v1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_write_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Refusing to overwrite frozen input: ' + str(path))
    else:
        write_json(path, value)

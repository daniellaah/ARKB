"""Registration inputs pinned while resources are unavailable; refuse any drift before freezing."""
import json

from arkb.evaluation.external import digest
from .common import ROOT


def working_files():
    names = {str(p.relative_to(ROOT)) for base in (ROOT / 'src', ROOT / 'evaluation/agentic_tools')
             for p in base.rglob('*.py')} | {'pyproject.toml'}
    return {name: digest(ROOT / name) for name in sorted(names)}


def verify_registration_inputs(out):
    pinned = json.loads((out / 'registration-inputs.json').read_text())
    actual_names = {str(p.relative_to(ROOT)) for base in (ROOT / 'src', ROOT / 'evaluation/agentic_tools')
                    for p in base.rglob('*.py')} | {'pyproject.toml'}
    if actual_names != set(pinned['working_files']):
        raise ValueError('Tested source file inventory changed before registration.')
    for name, checksum in pinned['working_files'].items():
        if digest(ROOT / name) != checksum:
            raise ValueError('Tested source changed before registration: ' + name)
    for name, checksum in pinned['prepared_files'].items():
        if digest(out / name) != checksum:
            raise ValueError('Prepared pilot input changed before registration: ' + name)



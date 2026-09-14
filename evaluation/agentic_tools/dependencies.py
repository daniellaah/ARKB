"""Fail before trials if the background environment cannot execute exact search."""
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from arkb.evaluation.external import digest
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval.exact import ExactRetriever


def dependency_identity():
    rg = shutil.which('rg')
    if rg is None:
        raise ValueError('ripgrep is unavailable on the worker PATH; no trials may start.')
    binary = Path(rg).resolve()
    version = subprocess.check_output([str(binary), '--version'], text=True, timeout=10)
    return {'path_environment': os.environ.get('PATH', ''), 'rg_path': str(binary),
            'rg_sha256': digest(binary), 'rg_version': version}


def verify_dependencies(expected=None):
    identity = dependency_identity()
    if expected is not None and identity != expected:
        raise ValueError('Background retrieval dependency identity changed.')
    with TemporaryDirectory(prefix='arkb-agentic-dependency-') as folder:
        directory = Path(folder)
        (directory / 'fixture.md').write_text('Orion release 2042.\nBeta release 2043.\n')
        exact = ExactRetriever(DocumentAccess(directory, vault_id='synthetic-dependency-check'))
        insensitive = exact.search('orion', case_sensitive=False).results
        regex = exact.search(r'Beta release \d{4}', regex=True).results
        if [r.content for r in insensitive] != ['Orion'] or [r.content for r in regex] != ['Beta release 2043']:
            raise ValueError('Production exact-search dependency smoke check failed.')
    return identity

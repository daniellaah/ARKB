import pytest
from pathlib import Path
import shutil

from .dependencies import verify_dependencies


def test_missing_launchd_path_fails_before_model_calls(monkeypatch, tmp_path):
    monkeypatch.setenv('PATH', str(tmp_path))
    with pytest.raises(ValueError, match='ripgrep is unavailable'):
        verify_dependencies()


def test_background_path_runs_production_casefold_and_regex(monkeypatch):
    rg = shutil.which('rg')
    if rg is None:
        pytest.skip('Real ripgrep is required for this integration check.')
    monkeypatch.setenv('PATH', str(Path(rg).parent) + ':/usr/bin:/bin:/usr/sbin:/sbin')
    actual = verify_dependencies()
    assert actual['rg_version'].startswith('ripgrep ')
    assert verify_dependencies(actual) == actual
    with pytest.raises(ValueError, match='identity changed'):
        verify_dependencies({**actual, 'rg_sha256': 'invalid'})

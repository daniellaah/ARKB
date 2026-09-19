"""Pilot v3 pipeline registration guards; no provider calls."""
import pytest

from arkb.evaluation.external import write_json


def test_registration_waits_for_training_without_provider_calls(tmp_path, monkeypatch):
    from evaluation.studies.agentic_pilot_v3 import pipeline as pilot_pipeline
    from evaluation.agentic_tools import preflight
    parent, out = tmp_path / 'parent', tmp_path / 'out'
    parent.mkdir()
    out.mkdir()
    write_json(parent / 'protocol.json', {})
    monkeypatch.setattr(pilot_pipeline, 'gpu_competitors', lambda: ['training process'])
    monkeypatch.setattr(preflight, 'main', lambda *a: pytest.fail('Provider must not run during training'))
    result = pilot_pipeline.register_if_available(out, parent, tmp_path)
    assert result['status'] == 'waiting_for_gpu'
    assert not (out / 'protocol.json').exists()


def test_registration_does_not_take_another_inference_owners_lock(tmp_path):
    import fcntl
    from evaluation.studies.agentic_pilot_v3.pipeline import register_if_available
    write_json(tmp_path / 'protocol.json', {})
    with (tmp_path / 'inference.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert register_if_available(tmp_path, tmp_path, tmp_path)['status'] == 'waiting_for_inference_lock'



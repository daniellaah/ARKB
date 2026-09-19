from copy import deepcopy
import json

import pytest

from arkb.evaluation.external import digest, write_json
from .contract import ARMS, OPTIONS
from .repeatability import selected_baseline, repeat_schedule, fingerprint
from .selection import attempt_key
from .summarize_pilot import summarize


def rows():
    result = []
    for unit, dataset in enumerate(('browsecomp-plus', 'fiqa', 'nfcorpus', 'musique')):
        for identifier in ('first', 'second'):
            for arm in ARMS:
                for variant in (('v0', 'v1') if dataset == 'musique' else ('v0',)):
                    result.append({'key': f'{dataset}-{identifier}-{arm.id}-{variant}',
                                   'schedule': {'dataset': dataset, 'id': identifier, 'arm': arm.id,
                                                'variant': variant, 'unit_index': unit, 'repetition': 0},
                                   'result': {'final': {'status': 'error'}}})
    return result


def test_all_arms_and_paired_variants_retained_without_quality_selection():
    source = rows()
    baseline = selected_baseline(source)
    assert len(baseline) == 35
    assert {r['schedule']['arm'] for r in baseline} == {a.id for a in ARMS}
    for row in source: row['result']['final']['status'] = 'answered'
    assert [r['key'] for r in selected_baseline(source)] == [r['key'] for r in baseline]


def test_seventy_new_runs_rotate_arms_and_preserve_mu_pairs():
    schedule = repeat_schedule(selected_baseline(rows()))
    assert len(schedule) == 70
    assert {s['repetition'] for s in schedule} == {1, 2}
    assert len({(s['dataset'], s['id'], s['variant'], s['arm'], s['repetition']) for s in schedule}) == 70
    assert schedule[0]['arm'] == ARMS[1].id
    assert schedule[7]['arm'] == ARMS[2].id
    mu = [s for s in schedule if s['dataset'] == 'musique']
    for a, b in zip(mu[::2], mu[1::2], strict=True):
        assert a['arm'] == b['arm'] and (a['variant'], b['variant']) == ('v0', 'v1')


def fingerprint_row(prefix, content='evidence'):
    one, ten = prefix + '_1', prefix + '_10'
    refs = {one: {'source': 'one.md', 'content': 'one'}, ten: {'source': 'ten.md', 'content': content}}
    return {'elapsed_ms': 1, 'evidence_sources': {'delivered': ['ten.md']}, 'result': {
        'final': {'answer': 'See ' + ten, 'evidence_refs': [ten]}, 'stop_reason': 'final',
        'observation': {'evidence_references': refs, 'models': [], 'tools': [
            {'executed': True, 'name': 'read', 'arguments': {'ref': ten}, 'status': 'success',
             'raw_result': {'result': {'ref': ten}}, 'delivered_to_conversation': True, 'submitted_to_model': True}]}}}


def test_known_ref_normalization_preserves_one_vs_ten_and_content_identity():
    a, b = fingerprint_row('ev_aaa'), fingerprint_row('ev_bbb')
    assert fingerprint(a) == fingerprint(b)
    changed = fingerprint(fingerprint_row('ev_bbb', content='different passage'))
    assert changed['evidence_identity_sha256'] != fingerprint(a)['evidence_identity_sha256']
    assert changed['final_sha256'] != fingerprint(a)['final_sha256']


def test_unknown_invalid_references_are_not_repaired_for_comparison():
    a, b = fingerprint_row('ev_aaa'), fingerprint_row('ev_aaa')
    b['result']['final']['evidence_refs'] = ['invented-ref']
    assert fingerprint(a)['final_sha256'] != fingerprint(b)['final_sha256']


@pytest.mark.parametrize('completed', [0, 2])
def test_accounting_uses_registered_count_and_does_not_project_core(tmp_path, completed):
    schedule = [{'dataset': 'fixture', 'id': str(i), 'variant': 'v0', 'arm': 'F-S', 'repetition': 1} for i in range(2)]
    protocol = {'phase': 'pilot', 'pilot_attempts': 2, 'files': {}, 'project_core_costs': False}
    for name, value in [('protocol.json', protocol), ('pilot-schedule.json', schedule), ('core-schedule.json', []),
                        ('design.json', {'options': OPTIONS, 'think': False})]:
        write_json(tmp_path / name, value)
    protocol_hash = digest(tmp_path / 'protocol.json')
    for item in schedule[:completed]:
        key = attempt_key(protocol_hash, item)
        directory = tmp_path / 'pilot-attempts' / key
        directory.mkdir(parents=True)
        row = {'key': key, 'protocol_sha256': protocol_hash, 'schedule': item, 'elapsed_ms': 1,
               'result': None, 'error': {'type': 'fixture', 'message': 'intentional recorded failure'}}
        write_json(directory / 'result.json', row)
        write_json(directory / 'attempt.json', {})
        (directory / 'provider.jsonl').write_text('')
        write_json(directory / 'complete.json', {'result_sha256': digest(directory / 'result.json'),
                                                'provider_sha256': digest(directory / 'provider.jsonl')})
    result = summarize(tmp_path)
    assert result['attempts_required'] == 2 and result['attempts_accounted'] == completed
    assert result['status'] == ('complete' if completed == 2 else 'partial')
    assert result['projected_core_inference_hours'] is None

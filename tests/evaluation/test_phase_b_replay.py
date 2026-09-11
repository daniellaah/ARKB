"""Public offline audit verifies frozen source sets, the untouched tail and gold isolation."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from arkb.evaluation.external import digest, write_json


def fixture_run(tmp_path):
    run = tmp_path/'run'; run.mkdir()
    (run/'pools').mkdir(); (run/'scoring').mkdir()
    before = [f'doc{i:02}' for i in reversed(range(20))] + ['doc20', 'doc21']
    baseline = sorted(before[:20]) + before[20:]
    # Single relevant source is first in Hybrid and twentieth in old identity-tied ranking.
    old = {'ndcg@10': 0., 'recall@10': 0., 'mrr@10': 0.,
           'ndcg@100': 0.227670248696953, 'recall@100': 1., 'mrr@100': .05}
    new = {k: 1. for k in old}
    for ds in ('scifact', 'bright-stackoverflow', 'bright-robotics'):
        aspects = {} if ds == 'scifact' else {'q': [{'id': 'a', 'weight': 1., 'supporting_docs': ['doc19']}]}
        additions = {} if not aspects else {'alpha_ndcg@10': 0., 'aspect_recall@10': 0.}
        row = {'qid': 'q', 'query': 'A fixed technical question?', 'hybrid_ranking': before,
               'candidate_document_ids': before[:20], 'baseline_scores': [1.]*20,
               'baseline_ranking': baseline, 'baseline_stage_ms': 20.}
        (run/'pools'/f'{ds}.jsonl').write_text(json.dumps(row)+'\n')
        write_json(run/'scoring'/f'{ds}.json', {'qrels': {'q': {'doc19': 1}}, 'aspects': aspects,
            'baseline_metrics': {'q': {**old, **additions}},
            'hybrid_metrics': {'q': {**new, **{k: 1. for k in additions}}}})
    write_json(run/'manifest.json', {'fixture': True})
    write_json(run/'checksums.json', {p.relative_to(run).as_posix(): digest(p)
               for p in run.rglob('*') if p.is_file()})
    return run, before, baseline


def audit(run, variant, output):
    root = Path(__file__).resolve().parents[2]
    return subprocess.run([sys.executable, str(root/'evaluation/audits/summarize_phase_b.py'),
                           str(run), '--variant', variant, '--output', str(output)],
                          capture_output=True, text=True)


def test_stable_score_replay_preserves_tail_and_recovers_known_positive(tmp_path):
    run, before, baseline = fixture_run(tmp_path)
    output = tmp_path/'audit'
    result = audit(run, 'B1', output)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output/'summary.json').read_text())
    for ds, row in summary['datasets'].items():
        assert row['metrics']['ndcg@10'] == row['metrics']['recall@100'] == 1.
        assert row['versus_B0']['ndcg@10'] == {'mean_delta': 1., 'ci95': [1., 1.], 'wins': 1, 'ties': 0, 'losses': 0}
        assert row['versus_hybrid']['recall@100']['mean_delta'] == 0
        replay = json.loads((output/f'{ds}-replay.jsonl').read_text())
        assert replay['ranking'] == before
        assert replay['ranking'][20:] == baseline[20:]


def test_audit_rejects_reordering_outside_the_rerank_pool(tmp_path):
    run, before, _ = fixture_run(tmp_path)
    trial = run/'bad'; trial.mkdir()
    (trial/'scifact.jsonl').write_text(json.dumps({'qid': 'q', 'scores': [1.]*20,
        'ranking': before[:20]+list(reversed(before[20:]))})+'\n')
    write_json(trial/'checksums.json', {'scifact.jsonl': digest(trial/'scifact.jsonl')})
    result = audit(run, 'bad', tmp_path/'audit')
    assert result.returncode != 0 and 'Candidate set or tail changed' in result.stderr


def test_audit_rejects_an_order_not_justified_by_saved_scores(tmp_path):
    run, _, baseline = fixture_run(tmp_path)
    trial = run/'bad'; trial.mkdir()
    for ds in ('scifact', 'bright-stackoverflow', 'bright-robotics'):
        (trial/f'{ds}.jsonl').write_text(json.dumps({'qid': 'q', 'scores': [1.]*20,
                                                   'ranking': baseline})+'\n')
    write_json(trial/'checksums.json', {p.name: digest(p) for p in trial.iterdir()})
    result = audit(run, 'bad', tmp_path/'audit')
    assert result.returncode != 0 and 'Scores do not justify' in result.stderr


@pytest.mark.parametrize('reason,scores,inputs', [
    ('all_equal_scores', list(range(20)), []),
    ('empty_body', [], [{'body_tokens_retained': 1}] * 20),
    ('scoring_error', [], []),
    ('unregistered_heuristic', [1.] * 20, []),
])
def test_audit_requires_evidence_for_fallback_trigger(tmp_path, reason, scores, inputs):
    run, before, _ = fixture_run(tmp_path)
    trial = run/'bad'; trial.mkdir()
    (trial/'scifact.jsonl').write_text(json.dumps({'qid': 'q', 'scores': scores,
        'ranking': before, 'fallback': reason, 'inputs': inputs})+'\n')
    write_json(trial/'checksums.json', {'scifact.jsonl': digest(trial/'scifact.jsonl')})
    result = audit(run, 'bad', tmp_path/'audit')
    assert result.returncode != 0 and 'Unsupported fallback trigger' in result.stderr


def test_exact_equal_fallback_is_audited_as_identity_without_new_scores(tmp_path):
    run, before, _ = fixture_run(tmp_path)
    trial = run/'B5'; trial.mkdir()
    for ds in ('scifact', 'bright-stackoverflow', 'bright-robotics'):
        (trial/f'{ds}.jsonl').write_text(json.dumps({'qid': 'q', 'scores': [1.] * 20,
            'ranking': before, 'fallback': 'all_equal_scores'})+'\n')
    write_json(trial/'checksums.json', {p.name: digest(p) for p in trial.iterdir()})
    output = tmp_path/'audit'
    result = audit(run, 'B5', output)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert all(d['fallback_activation_rate'] == 1. for d in summary['datasets'].values())


@pytest.mark.parametrize('values,expected', [
    ({'B3': [.77, .6, .7, .5, .6], 'B4': [.77, .55, .65, .45, .55]}, 'B3'),
    ({'B3': [.77, .6, .7, .45, .55], 'B4': [.77, .55, .65, .5, .6]}, 'B4'),
    ({'B3': [.70, .8, .9, .8, .9], 'B4': [.77, .55, .65, .5, .6]}, 'B4'),
    ({'B3': [.70, .8, .9, .8, .9], 'B4': [.70, .8, .9, .8, .9]}, 'B2'),
])
def test_selection_cli_applies_one_registered_policy_across_domains(tmp_path, values, expected):
    run, _, _ = fixture_run(tmp_path)
    allocation = run/'allocations'; allocation.mkdir()
    write_json(allocation/'protocol.json', {'fixture': 'registered shared selection'})
    write_json(allocation/'checksums.json', {'protocol.json': digest(allocation/'protocol.json')})
    datasets = ('scifact', 'bright-stackoverflow', 'bright-robotics')
    for variant in ('B0', 'B1', 'B2', 'B3', 'B4'):
        folder = run/'final-analyses'/variant; folder.mkdir(parents=True)
        vector = values.get(variant, [.77, .4, .5, .35, .4])
        summaries = {}
        for ds, ndcg, aspect in zip(datasets, [vector[0], vector[1], vector[3]],
                                    [None, vector[2], vector[4]]):
            m = {'ndcg@10': ndcg}
            if aspect is not None: m['aspect_recall@10'] = aspect
            summaries[ds] = {'metrics': m, 'versus_hybrid': {
                k: {'mean_delta': 0., 'ci95': [-.01, .01]} for k in m}}
            (folder/f'{ds}-attribution.jsonl').write_text(json.dumps({'qid': 'q', 'metrics': m})+'\n')
        write_json(folder/'summary.json', {'pool_manifest_sha256': digest(run/'manifest.json'),
                                          'datasets': summaries})
        write_json(folder/'checksums.json', {p.name: digest(p) for p in folder.iterdir()})
    root = Path(__file__).resolve().parents[2]
    output = tmp_path/'selection'
    result = subprocess.run([sys.executable, str(root/'evaluation/audits/select_phase_b.py'),
                             str(run), '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    decision = json.loads((output/'decision.json').read_text())
    assert decision['selected_variant'] == expected
    assert decision['recommendation'] == 'reranking should remain optional'
    assert decision['default_quality_gate_passed'] is False
    assert decision['paired_comparisons']['B2_versus_B1']['scifact']['ndcg@10']['mean_delta'] == 0


def test_legacy_score_replay_inherits_verified_input_diagnostics(tmp_path):
    run, before, _ = fixture_run(tmp_path)
    control = run/'B2-corrected'; control.mkdir()
    fields = {'query_tokens_before': 1000, 'query_tokens_retained': 464,
              'title_tokens_before': 0, 'title_tokens_retained': 0,
              'body_tokens_before': 100, 'body_tokens_retained': 0, 'truncated': True}
    for ds in ('scifact', 'bright-stackoverflow', 'bright-robotics'):
        (control/f'{ds}.jsonl').write_text(json.dumps({'qid': 'q', 'scores': [1.]*20,
            'ranking': before, 'inputs': [fields]*20})+'\n')
    write_json(control/'checksums.json', {p.name: digest(p) for p in control.iterdir()})
    output = tmp_path/'audit'
    result = audit(run, 'B1', output)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output/'summary.json').read_text())
    assert summary['legacy_input_diagnostics_sha256'] == digest(control/'checksums.json')
    for row in summary['datasets'].values():
        assert row['inputs']['zero_body_rate'] == 1. and row['inputs']['all_body_empty_queries'] == 1
        assert row['metrics']['ndcg@10'] == 1.

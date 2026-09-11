"""Public offline audit verifies frozen source sets, the untouched tail and gold isolation."""

import json
from pathlib import Path
import subprocess
import sys

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

"""Post-hoc diagnostics of frozen validation rankings; no retrieval or tuning."""

import argparse
from datetime import datetime, timezone
import gzip
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from arkb.evaluation.external import digest, rank_metrics, reference_metrics, write_json


ARMS = ("bm25", "semantic", "C0")
KS = (10, 20, 100)
METRICS = ("ndcg@10", "recall@10", "recall@20", "recall@100")


def read_json(path):
    return json.loads(path.read_text())


def read_trec(path, query_ids, arm):
    rankings = {qid: [] for qid in query_ids}
    with path.open() as stream:
        for line in stream:
            qid, placeholder, doc, rank, score, tag = line.split()
            assert qid in rankings and placeholder == "Q0" and tag == "ARKB-" + arm
            assert int(rank) == len(rankings[qid]) + 1 and float(score) > 0
            rankings[qid].append(doc)
    for ranking in rankings.values():
        assert len(ranking) == len(set(ranking))
        assert len(ranking) <= (1000 if arm == "C0" else 500)
    return rankings


def partition(positives, rankings, k):
    union = set(rankings["bm25"]) | set(rankings["semantic"])
    assert set(rankings["C0"]) == union
    found = positives & set(rankings["C0"][:k])
    below = (positives & union) - found
    absent = positives - union
    assert found | below | absent == positives
    assert len(found) + len(below) + len(absent) == len(positives)
    total = len(positives)
    assert total > 0
    oracle = min(k, len(positives & union)) / total
    return {
        "relevant_total": total,
        "in_C0_top_k": len(found),
        "in_union_below_C0_top_k": len(below),
        "absent_from_union": len(absent),
        "recall": len(found) / total,
        "union_coverage": len(positives & union) / total,
        "union_below_top_k_fraction": len(below) / total,
        "absent_fraction": len(absent) / total,
        "oracle_recall_at_k": oracle,
        "oracle_minus_actual_recall": oracle - len(found) / total,
        "unavoidable_cutoff_fraction": max(0, len(positives & union) - k) / total,
    }


def paired(values, seed, repetitions):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    # Bounded memory and identical resamples for all metrics in this dataset.
    samples = []
    for start in range(0, repetitions, 128):
        indices = rng.integers(0, len(values), size=(min(128, repetitions - start), len(values)))
        samples.extend(values[indices].mean(axis=1).tolist())
    return {
        "mean_delta": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()),
    }


def analyze(name, run, data, evidence, output, seed, repetitions):
    replay = read_json(evidence / "validation" / f"{name}-replay.json")
    assert replay["status"] == "passed"
    assert digest(run / "checksums.json") == replay["run_checksums_sha256"]
    run_checks = read_json(run / "checksums.json")
    data_checks = read_json(data / "checksums.json")
    consumed = {}
    for base, checks, names in (
        (run, run_checks, [*(arm + ".trec" for arm in ARMS), "summary.json", "protocol.json",
                           "experiment.json", "development-decision.json", "data-manifest.json"]),
        (data, data_checks, ["manifest.json", "queries.json", "qrels.json"]),
    ):
        for filename in names:
            path = base / filename
            actual = digest(path)
            assert actual == checks[filename], path
            consumed[str(path)] = actual
    protocol = read_json(run / "protocol.json")
    experiment = read_json(run / "experiment.json")
    original = read_json(run / "summary.json")
    manifest = read_json(data / "manifest.json")
    qrels = read_json(data / "qrels.json")
    queries = read_json(data / "queries.json")
    ids = protocol["query_ids"]
    assert experiment["status"] == "completed"
    assert read_json(run / "development-decision.json")["selected_policy"] == "C0"
    assert experiment["snapshot_before"] == experiment["snapshot_after"]
    assert protocol["dataset_manifest_sha256"] == experiment["dataset_manifest_sha256"] == digest(data / "manifest.json")
    assert digest(run / "data-manifest.json") == digest(data / "manifest.json")
    assert ids == manifest["query_ids"] == list(queries)
    assert set(ids) == set(qrels) and len(ids) == original["query_count"]
    assert protocol["leg_depth"] == 500 and protocol["rrf_k"] == 60 and not protocol["rerank"]
    rankings = {arm: read_trec(run / (arm + ".trec"), ids, arm) for arm in ARMS}
    metrics = {arm: {qid: rank_metrics(qrels[qid], ranks, ks=KS)
                     for qid, ranks in rankings[arm].items()} for arm in ARMS}
    summary_error = reference_error = 0.0
    for arm in ARMS:
        reference = reference_metrics(qrels, rankings[arm], ks=KS)
        for qid in ids:
            for metric, value in reference[qid].items():
                reference_error = max(reference_error, abs(value - metrics[arm][qid][metric]))
        for metric, expected in original["arms"][arm]["labels"]["qrels"]["metrics"].items():
            actual = float(np.mean([metrics[arm][qid][metric] for qid in ids]))
            summary_error = max(summary_error, abs(expected - actual))
    assert summary_error < 1e-12 and reference_error < 1e-12
    rows = []
    for qid in ids:
        positives = {doc for doc, grade in qrels[qid].items() if grade > 0}
        ranks = {arm: rankings[arm][qid] for arm in ARMS}
        positions = {arm: {doc: i for i, doc in enumerate(ranks[arm], 1)} for arm in ARMS}
        semantic_top = positives & set(ranks["semantic"][:10])
        hybrid_top = positives & set(ranks["C0"][:10])
        row = {
            "qid": qid,
            "metrics": {arm: metrics[arm][qid] for arm in ARMS},
            "C0_minus_semantic": {m: metrics["C0"][qid][m] - metrics["semantic"][qid][m] for m in METRICS},
            "cutoffs": {str(k): partition(positives, ranks, k) for k in KS},
            "semantic_top10_relevant_lost": sorted(semantic_top - hybrid_top),
            "C0_top10_relevant_added": sorted(hybrid_top - semantic_top),
            "bm25_exclusive_relevant": sorted((positives & set(ranks["bm25"])) - set(ranks["semantic"])),
            "semantic_exclusive_relevant": sorted((positives & set(ranks["semantic"])) - set(ranks["bm25"])),
            "positive_positions": {doc: {"grade": qrels[qid][doc], **{arm: positions[arm].get(doc) for arm in ARMS}}
                                   for doc in sorted(positives)},
        }
        for k in KS:
            value = row["cutoffs"][str(k)]
            assert abs(value["recall"] - metrics["C0"][qid][f"recall@{k}"]) < 1e-14
            assert abs(value["recall"] + value["oracle_minus_actual_recall"] +
                       value["unavoidable_cutoff_fraction"] + value["absent_fraction"] - 1) < 1e-14
        rows.append(row)
    per_query = output / f"{name}-per-query.jsonl.gz"
    with per_query.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
        for row in rows:
            zipped.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
    result = {
        "dataset": name, "queries": len(ids), "corpus_count": manifest["corpus_count"],
        "mean_metrics": {arm: {m: original["arms"][arm]["labels"]["qrels"]["metrics"][m] for m in METRICS} for arm in ARMS},
        "paired_C0_minus_semantic": {m: paired([r["C0_minus_semantic"][m] for r in rows], seed, repetitions) for m in METRICS},
        "cutoffs": {str(k): {"macro": {key: float(np.mean([r["cutoffs"][str(k)][key] for r in rows]))
                                         for key in rows[0]["cutoffs"][str(k)]},
                                "pooled_query_document_counts": {key: sum(r["cutoffs"][str(k)][key] for r in rows)
                                    for key in ("relevant_total", "in_C0_top_k", "in_union_below_C0_top_k", "absent_from_union")}}
                    for k in KS},
        "movements": {key: {"queries": sum(bool(r[key]) for r in rows), "query_document_pairs": sum(len(r[key]) for r in rows)}
                      for key in ("semantic_top10_relevant_lost", "C0_top10_relevant_added", "bm25_exclusive_relevant", "semantic_exclusive_relevant")},
        "queries_with_zero_union_positives": sum(r["cutoffs"]["10"]["union_coverage"] == 0 for r in rows),
        "queries_with_union_positive_but_zero_C0_top10": sum(r["cutoffs"]["10"]["union_coverage"] > 0 and
                                                             r["cutoffs"]["10"]["recall"] == 0 for r in rows),
        "illustrative_query_ids": {
            "largest_ndcg_losses": [r["qid"] for r in sorted(rows, key=lambda r: (r["C0_minus_semantic"]["ndcg@10"], r["qid"]))[:3]],
            "largest_ndcg_gains": [r["qid"] for r in sorted(rows, key=lambda r: (-r["C0_minus_semantic"]["ndcg@10"], r["qid"]))[:3]],
        },
        "verification": {"summary_metric_max_error": summary_error, "trec_eval_max_error": reference_error,
                         "reference_metric_checks": len(ids) * len(ARMS) * 6,
                         "partition_checks": len(ids) * len(KS), "consumed_files_sha256": consumed,
                         "prior_replay_sha256": digest(evidence / "validation" / f"{name}-replay.json"),
                         "run_checksums_sha256": digest(run / "checksums.json")},
    }
    # Confirm read-only inputs remained stable across this analysis.
    assert all(digest(Path(path)) == expected for path, expected in consumed.items())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = root / "evaluation/phase-c/v1"
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "analysis": "post-hoc descriptive validation diagnostics; aggregate results already observed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {"nfcorpus": "nfcorpus-r2", "fiqa": "fiqa-r2"},
        "arms": list(ARMS), "cutoffs": list(KS), "paired_metrics": list(METRICS),
        "bootstrap": {"seed": 20260912, "repetitions": 10000, "unit": "query within dataset",
                      "interval": "exploratory paired percentile 95%; no multiplicity correction",
                      "ties": "exact equality", "numpy": np.__version__},
        "model_calls": 0, "retrieval_calls": 0, "policy_changes": 0,
        "case_selection": "three largest nDCG losses and gains, then query ID; illustrative, not representative",
        "scope": "All frozen queries and original qrels; existing TREC ranks only. No corpus/query reduction or retuning.",
        "limitations": ["Public validation labels already exposed; not a new held-out experiment.",
                        "Unjudged documents are unknown, not established negatives.",
                        "Source labels do not establish representative chunk adequacy or causal mechanism.",
                        "Union means captured 500-chunk legs, not exhaustive corpus recall.",
                        "Oracle sorting uses labels only as a diagnostic bound, not a deployable policy.",
                        "FiQA blank-document exception and all original relevance denominators remain unchanged."],
        "script_sha256": digest(__file__),
        "metric_module_sha256": digest(root / "src/arkb/evaluation/external.py"),
        "pytrec_eval_terrier": importlib.metadata.version("pytrec-eval-terrier"),
    }
    write_json(args.output / "protocol.json", protocol)
    # Hand-computable mixed loss fixture, independent of real benchmark outcomes.
    fixture = partition(set("abcde"), {"bm25": list("axbc"), "semantic": list("ydab"), "C0": list("xyadbc")}, 2)
    assert (fixture["in_C0_top_k"], fixture["in_union_below_C0_top_k"], fixture["absent_from_union"]) == (0, 4, 1)
    assert fixture["oracle_recall_at_k"] == 0.4 and fixture["unavoidable_cutoff_fraction"] == 0.4
    assert paired([0, 0, 0], 1, 128)["ci95"] == [0.0, 0.0]
    summary = {"status": "completed", "protocol_sha256": digest(args.output / "protocol.json"), "datasets": {}}
    for name, run_name in protocol["datasets"].items():
        result = analyze(name, args.storage / "validation" / run_name, args.storage / "data" / name,
                         evidence, args.output, protocol["bootstrap"]["seed"], protocol["bootstrap"]["repetitions"])
        summary["datasets"][name] = result
        print(name, result["paired_C0_minus_semantic"]["ndcg@10"], flush=True)
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "checksums.json", {p.name: digest(p) for p in sorted(args.output.iterdir())})


if __name__ == "__main__":
    main()

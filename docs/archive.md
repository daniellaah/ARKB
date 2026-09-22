# Archive index

Reports, plans and protocols of completed work, kept for provenance under
`docs/archive/`. Their code was removed from the working tree in commit
`f36e682` (2026-09-21); the tree as it stood just before, with every module
these documents describe, is commit `f70ae8e`. Paths inside the archived
documents refer to that tree. Frozen artifacts they cite live under
`archive/` (moved unchanged in `f36e682`); registered runs and their source
snapshots stay on `/Volumes/ARKBPhaseC`.

Restore any removed module with, for example:

```bash
git checkout f70ae8e -- evaluation/agentic_tools src/arkb/evaluation
```

## Evaluation package v2 (2026-09-21, superseded the same day)

[eval-design-v2.md](archive/eval-design-v2.md) designed a seven-slice
development evaluation (v2 + exact-v2 in repo, five optional slices on the
external volume, span-level scorer, bootstrap comparison, blinded review
tooling); it was built at `f70ae8e`..`d83ef6b` and then collapsed into the
single-script evaluation that stands now. Comparison tables of the
agentic-versus-workflow runs and the fix-first log are under
`archive/comparisons/`.

## Agentic evaluation studies (2026-09-12 to 2026-09-20)

Code: `evaluation/agentic_tools` (library), `evaluation/studies` (pipelines),
`evaluation/audits` (replay audits). Artifacts: `archive/agentic-tools/<run>/`,
`archive/core-intake/`. Handoff record: `evaluation/results/agentic-tools-v1/orchestration-status.json` (local).

| Document | What it is | Last code commit |
| --- | --- | --- |
| [agentic-tool-selection-evaluation-plan.md](archive/agentic-tool-selection-evaluation-plan.md) | seven-arm study design, plan of record | befa4d4 |
| [agentic-tool-selection-core-protocol.md](archive/agentic-tool-selection-core-protocol.md) | registered core v1 protocol | befa4d4 |
| [agentic-tool-selection-pilot-amendment-v2.md](archive/agentic-tool-selection-pilot-amendment-v2.md), [v3](archive/agentic-tool-selection-pilot-amendment-v3.md) | pilot amendments | 0d5655d |
| [agentic-tool-selection-execution.md](archive/agentic-tool-selection-execution.md) | execution notes | 0d5655d |
| [agentic-pilot-v3-results.md](archive/agentic-pilot-v3-results.md) | pilot v3 results (audited) | 0d5655d |
| [agentic-serving-probe-v1.md](archive/agentic-serving-probe-v1.md), [agentic-serving-v1-results.md](archive/agentic-serving-v1-results.md) | serving concurrency probe and results | 0d5655d |
| [agentic-repeatability-v1.md](archive/agentic-repeatability-v1.md), [agentic-repeatability-v1-results.md](archive/agentic-repeatability-v1-results.md) | repeatability study: two thirds of cells change their final between identical runs | 0d5655d |
| [tool-overhead-repair.md](archive/tool-overhead-repair.md) | tool-overhead repair verified on a development replay | befa4d4 |
| [agentic-tool-selection-design-review.md](archive/agentic-tool-selection-design-review.md) | why core v1 was stopped | 0d5655d |
| [agentic-core-v2-protocol.md](archive/agentic-core-v2-protocol.md) | registered core v2 design, stopped after 16 attempts | 0d5655d |

The agentic-versus-workflow comparison that followed stays current:
[agile-vs-workflow-v1.md](agile-vs-workflow-v1.md) (its arm adapters,
`evaluation/agentic_tools/contract.py`, were removed with the library; its
run records stay under `evaluation/results/devloop/aw-*`, local only).

## Retrieval-layer phases A to C (2026-09-11 to 2026-09-13)

Code: `evaluation/experiments`. Artifacts: `archive/phase-a/`, `archive/phase-b/`, `archive/phase-c/`.

| Document | What it is | Last code commit |
| --- | --- | --- |
| [phase-a-plan.md](archive/phase-a-plan.md), [phase-a-report.md](archive/phase-a-report.md) | tool contract and reliability | 00e9bb7 |
| [phase-b-plan.md](archive/phase-b-plan.md), [phase-b-report.md](archive/phase-b-report.md) | reranker | 84d2735 |
| [phase-c-plan.md](archive/phase-c-plan.md), [phase-c-report.md](archive/phase-c-report.md), [phase-c-validation-diagnostics.md](archive/phase-c-validation-diagnostics.md) | candidate generation and fusion, full BrowseComp-Plus index | befa4d4 |
| [phase-c-execution-note.md](archive/phase-c-execution-note.md), [phase-c-scale-note.md](archive/phase-c-scale-note.md), [phase-c-storage-note.md](archive/phase-c-storage-note.md), [phase-c-index-preparation-reuse.md](archive/phase-c-index-preparation-reuse.md) | execution, scale, storage and index-reuse notes | befa4d4 |
| [embedding-throughput-investigation.md](archive/embedding-throughput-investigation.md) | embedding throughput | c140b2e |

## Early agent evaluation, P0 to P4 (2026-09-09 to 2026-09-11)

Code: `src/arkb/evaluation` (datasets, metrics, baselines, replay, external
adapters) and its tests under `tests/evaluation`. Artifacts:
`archive/reviews/`, `archive/metric-spec.json`.

| Document | What it is | Last code commit |
| --- | --- | --- |
| [eval-optimization-plan.md](archive/eval-optimization-plan.md), [implementation-progress.md](archive/implementation-progress.md) | the v2 evaluation plan and its execution status | ea6e395 |
| [public-dataset-selection.md](archive/public-dataset-selection.md) | which public datasets were adopted | ea6e395 |
| [p0-current-baseline-results.md](archive/p0-current-baseline-results.md), [p1-pilot-results.md](archive/p1-pilot-results.md), [p2-budget-results.md](archive/p2-budget-results.md), [p3-controlled-results.md](archive/p3-controlled-results.md), [p4-external-results.md](archive/p4-external-results.md) | P0 to P4 results | ea6e395 |
| [deterministic-retrieval-baselines.md](archive/deterministic-retrieval-baselines.md), [deterministic-baseline-results.md](archive/deterministic-baseline-results.md) | engine-only baselines | 7bbf4ce |
| [agent-vs-baselines.md](archive/agent-vs-baselines.md), [agent-model-ablation.md](archive/agent-model-ablation.md), [phase1-agent-model-ablation-results.md](archive/phase1-agent-model-ablation-results.md) | agent against baselines, model ablation | 66df891 |
| [reranker-integration.md](archive/reranker-integration.md) | reranker integration | 7bbf4ce |

## Product development notes (2026-09-09 to 2026-09-11)

| Document | What it is |
| --- | --- |
| [agent-thinking-validation.md](archive/agent-thinking-validation.md), [agent-search-stopping-validation.md](archive/agent-search-stopping-validation.md) | thinking and search-stopping validation |
| [refactor-results.md](archive/refactor-results.md), [simplification-results.md](archive/simplification-results.md), [directory-refactor-plan.md](archive/directory-refactor-plan.md) | refactors |
| [release-v0.1.0.md](archive/release-v0.1.0.md) | the 0.1.0 release |
| [english-scope-cleanup.md](archive/english-scope-cleanup.md) | English retrieval scope |

# Project map

Start here to learn the codebase. ARKB is an agentic retrieval system over a
flat directory of Markdown notes: a small LLM chooses among `match`, `search`
and `read`, and a deterministic retrieval engine executes them. Everything
below the agent is reproducible and label-free; everything about answer
quality is measured, not assumed.

## Reading order

1. [README.md](../README.md): what the product does and how to run it.
2. `src/arkb/agent/tools.py` (tool contract), `session.py` (validated boundary,
   evidence references), `observation.py` (budgets and accounting),
   `loop.py` (the bounded tool-calling loop).
3. `src/arkb/retrieval/`: `exact.py` (literal matching with a disposable text
   cache), `bm25.py`, `semantic.py`, `hybrid.py` (reciprocal rank fusion),
   `engine.py` (explicit mode dispatch).
4. `src/arkb/knowledge/`: documents, chunking, embeddings, SQLite snapshots,
   Qdrant collections, the index builder.
5. `src/arkb/runtime.py`: composition of services, engines and tools;
   `interfaces/cli.py`: the five commands.
6. [docs/arkb-current-system-review.md](arkb-current-system-review.md): the
   long-form architecture review and the evidence behind each design choice.

## Directory map

| Path | Role |
| --- | --- |
| `src/arkb/{agent,retrieval,knowledge,generation,interfaces}`, `runtime.py`, `config.py` | Product code |
| `src/arkb/evaluation/` | Evaluation library that ships with the package: datasets, metrics, baselines, replay, external benchmark adapters. Candidate to move out of the product package. |
| `evaluation/agentic_tools/` | Reusable library for registered agent studies: capability arms, transport, readiness gates, stop controller, selection, records, scoring, statistics, judge, analysis, runner, registration |
| `evaluation/studies/<study>/` | One-off preparation and pipeline scripts of each registered study, frozen history with a README each |
| `evaluation/` (package) | Development evaluation: devset (84 core + 94 optional scenarios), runner, scorer, paired comparison, blinded review, engine-only recall; see [docs/eval-design.md](eval-design.md) |
| `evaluation/audits/` | Independent replay audits of completed runs |
| `evaluation/experiments/` | Historical experiment scripts (P0 to P4, Phase A to C) |
| `evaluation/agentic-tools/<run>/` | Small public copies of registered run artifacts (protocols, accounting, audits) |
| `evaluation/results/` (ignored) | Local run outputs |
| `/Volumes/ARKBPhaseC/` | External volume: corpora, indexes, frozen study runs with their source snapshots |
| `tests/` | Deterministic suite (no services); `integration`-marked tests need Ollama and Qdrant |
| `docs/` | Reports and protocols; see the index below |

## Evaluation index

Current and active:

- [agile-vs-workflow-v1.md](agile-vs-workflow-v1.md): fully agentic against fixed-workflow retrieval on the development set (four arms, 9B with thinking, paired intervals; human review pending).
- [devloop-fix-log.md](devloop-fix-log.md): the fix-first development log with baseline and after measurements.
- [eval-design.md](eval-design.md) and [evaluation/README.md](../evaluation/README.md): the development evaluation, its design and how to run and compare runs.
- [agentic-tool-selection-evaluation-plan.md](agentic-tool-selection-evaluation-plan.md): the seven-arm study design (plan of record).
- [agentic-tool-selection-design-review.md](agentic-tool-selection-design-review.md): why core-v1 was stopped and what a replacement needs.
- [agentic-core-v2-protocol.md](agentic-core-v2-protocol.md): the registered core v2 design (stopped after 16 attempts).
- `evaluation/results/agentic-tools-v1/orchestration-status.json`: the durable handoff record for automation.

Completed development studies (audited):
[agentic-pilot-v3-results.md](agentic-pilot-v3-results.md),
[agentic-serving-v1-results.md](agentic-serving-v1-results.md),
[agentic-repeatability-v1-results.md](agentic-repeatability-v1-results.md),
[tool-overhead-repair.md](tool-overhead-repair.md).

Retrieval-layer phases (completed): [phase-a-report.md](phase-a-report.md)
(tool contract), [phase-b-report.md](phase-b-report.md) (reranker),
[phase-c-report.md](phase-c-report.md) (candidate generation and fusion, full
BrowseComp-Plus index) with
[phase-c-validation-diagnostics.md](phase-c-validation-diagnostics.md).

Historical (superseded, kept for provenance): the P0 to P4 result notes under
`evaluation/*.md`, [agent-thinking-validation.md](agent-thinking-validation.md),
[agent-search-stopping-validation.md](agent-search-stopping-validation.md),
[refactor-results.md](refactor-results.md), [simplification-results.md](simplification-results.md),
[directory-refactor-plan.md](directory-refactor-plan.md), [release-v0.1.0.md](release-v0.1.0.md),
[english-scope-cleanup.md](english-scope-cleanup.md), [embedding-throughput-investigation.md](embedding-throughput-investigation.md),
and the pilot amendments v2 and v3.

## Conventions

- Labels, answers and judge prompts never enter inference code paths; scoring
  reads them afterwards from separate files.
- Completed attempts are immutable and checksummed; interrupted work needs
  explicit accounting; nothing is retried silently.
- Every registered run freezes its own source snapshot; moving code in the
  working tree never changes a registered protocol hash.
- Numbers are reported with their denominators and limits; single-model,
  exposed-data development measurements are not quality claims.

## Everyday commands

`make test`, `make lint`, `make devset`, `make eval LABEL=<name>`, `make rescore LABEL=<name>`,
`make compare A=<run> B=<run>`; see `make help`.

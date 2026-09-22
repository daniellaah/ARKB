# Project map

Start here to learn the codebase. ARKB is an agentic retrieval system over a
directory tree of Markdown notes: a small LLM chooses among `match`, `search`
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
4. `src/arkb/knowledge/`: documents, chunking, embeddings, the wikilink graph,
   SQLite snapshots, Qdrant collections, the index builder.
5. `src/arkb/runtime.py`: composition of services, engines and tools;
   `agent/transports.py`: which chat transport a model name selects, and what a
   run cost; `interfaces/cli.py`: the seven commands; `interfaces/chat.py`: the
   multi-turn session as a function over input lines; `interfaces/mcp_server.py`:
   the same tools without `finish`, served read-only to an outer agent over MCP stdio.
6. [docs/arkb-current-system-review.md](arkb-current-system-review.md): the
   long-form architecture review and the evidence behind each design choice.

## Directory map

| Path | Role |
| --- | --- |
| `src/arkb/{agent,retrieval,knowledge,generation,interfaces}`, `runtime.py`, `config.py` | Product code |
| `evaluation/` | Development evaluation: 58 notes, 84 questions, one script (run, rescore, compare); see [evaluation/README.md](../evaluation/README.md) |
| `archive/` | Frozen artifacts of completed studies (phase A to C bundles, review sheets, registered-run protocols and accounting, comparison tables); evidence for the archived reports, never edited |
| `evaluation/results/` (ignored) | Local run outputs |
| `/Volumes/ARKBPhaseC/` | External volume: corpora, indexes, frozen study runs with their source snapshots |
| `tests/` | Deterministic suite (no services); `integration`-marked tests need Ollama and Qdrant |
| `docs/` | Reports and protocols; see the index below |

## Evaluation index

Current:

- [evaluation/README.md](../evaluation/README.md): the development evaluation (questions, metrics, commands).
- [devloop-fix-log.md](devloop-fix-log.md): the fix-first development log with baseline and after measurements, the model-capacity axis and the transport correction.
- [agile-vs-workflow-v1.md](agile-vs-workflow-v1.md): fully agentic against fixed-workflow retrieval on the development set (four arms, 9B with thinking, paired intervals; human review pending).
- [search-agent-roadmap.md](search-agent-roadmap.md): what the search agent still lacks, in priority order, with one executable prompt per task.
- [arkb-current-system-review.md](arkb-current-system-review.md): the long-form architecture review.

Completed work (studies, phases, early evaluation, product notes) is indexed
in [archive.md](archive.md) with the commit that last carried its code; the
documents live under `docs/archive/` and their artifacts under `archive/`.

## Conventions

- A source is a vault-relative POSIX path; absolute paths, backslashes, drive
  letters and `..` segments are rejected at every boundary that accepts one.
- Labels, answers and judge prompts never enter inference code paths; scoring
  reads them afterwards from separate files.
- Completed attempts are immutable and checksummed; interrupted work needs
  explicit accounting; nothing is retried silently.
- Every registered run freezes its own source snapshot; moving code in the
  working tree never changes a registered protocol hash.
- Numbers are reported with their denominators and limits; single-model,
  exposed-data development measurements are not quality claims.

## Everyday commands

`make test`, `make lint`, `make eval LABEL=<name>`, `make rescore LABEL=<name>`,
`make compare A=<run> B=<run>`; see `make help`.

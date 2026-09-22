# Agentic pilot amendment v2

Registered on 2026-09-13 America/Los_Angeles, after the original 98-attempt pilot
and before any core inference or benchmark answer grading. This amendment uses
the execution-failure amendment procedure in section 6 of the original plan.
It preserves the original plan and all first-pilot attempts.

## Evidence and decision

The original pilot finished at 2026-09-14T02:30:28Z. Its protocol SHA-256 is
`c8c0a5842769e1067ad989d1b363ec5ae0484d15e021133a58a388a86266da49`.
All 98 attempt identities, result/provider checksums and 1,262 trace checks
passed. There were 23 execution errors:

| Observed failure | Attempts | Interpretation |
| --- | ---: | --- |
| Empty final content | 11 | Provider returned no canonical answer; HTTP success is not answer success. |
| Output length limit | 4 | Generation exhausted the registered 4,096-token ceiling. |
| Invalid final reference | 5 | Actual model citation failure; retain it as an outcome. |
| Missing `rg` executable | 3 | Background environment defect affecting matching; repair before comparisons. |

The launchd job did not inherit the interactive shell's Homebrew PATH. A-M calls
using regex or case-insensitive matching raised `FileNotFoundError: rg`. The
replacement launch environment explicitly includes `/opt/homebrew/bin` and
pins the resolved ripgrep binary, its SHA-256 and version. Before any trial, the
worker validates that identity and runs both affected paths through the actual
production `ExactRetriever` on synthetic documents. Missing dependencies abort
before inference rather than becoming an arm's measured disadvantage.

Sixteen separate synthetic finalization requests used four predeclared contexts,
two repetitions and both thinking settings. No benchmark question, answer,
evidence or relevance label entered those requests. Thinking-enabled requests
produced valid canonical outputs in 6/8 cases; both repetitions of a fixed-RAG
missing-fact case exhausted 4,096 tokens. Nonthinking requests passed in 8/8
cases. This reproduces a bounded-output reliability problem; it does not prove
that every empty final has the same cause or establish answer-quality benefits.

For the replacement pilot, set **`think=False` for every request in all seven
arms**, retaining the same exact Qwen3.5-4B weights, temperature, context/output
limits, evidence/tool/turn/deadline ceilings, prompts and retrieval configuration.
The common adapter records and enforces this setting. The inference study is
therefore explicitly about the registered **nonthinking Qwen3.5-4B** configuration;
its eventual conclusions must not be generalized to the original thinking mode.

Ollama's [v0.33.2 chat handler](https://github.com/ollama/ollama/blob/v0.33.2/server/routes.go)
defers structured-output constraints while thinking and may internally restart
generation when content begins. For a nonthinking built-in parser it applies
constraints immediately. This source evidence motivates the workaround, but
the local probes and full replacement pilot determine whether it is operationally
adequate. Do not extract an answer from a thinking field or repair model output.

## Replacement run and gates

Run all **98 attempts again**, in the original frozen order and on the exact
same selected scenarios and complete corpora. Results live in a new directory,
`/Volumes/ARKBPhaseC/agentic-tools-v1/pilot-v2`. Reuse all 66 original MuSiQue READY
indexes and their original contexts. No document embeddings or full indexes are
rebuilt. Preserve the first pilot separately; do not combine its outcomes with
v2, selectively retry errors or drop failed cases from a denominator.

The replacement has a distinct source snapshot, executable protocol, dependency
identity, provider preflight, attempt keys and one-shot launchd job. Both revisions
share the original inference lock. Its status now refreshes the actual scenario
on every attempt. Context SQLite hashes are verified before access. None of
these changes modify production Agent or retrieval code, or Phase A/B/C results.

The original pilot projected 53.10 hours and 1.684 GiB of raw attempt artifacts
for 5,460 core trajectories under its failing configuration. This is historical
accounting, not a completion estimate for v2. Recalculate after all replacement
attempts; exclude index setup, grader inference, independent review and archival
from that inference estimate and report their costs separately.

Core dispatch still requires complete v2 accounting, a technical review of every
failure and a separate frozen core protocol. Citation errors and legitimate
budget stops remain measured outcomes; unresolved environmental defects or
systematic provider contract failures cannot be hidden in aggregate scores.
Judge calibration and independent review remain pending; no answer-quality
scores were used to select this amendment.

The synthetic probe initially failed during import because an optional validator
was unavailable. It made no model requests. The executable probe uses ARKB's
existing strict validator; no package was installed to bypass that failure.

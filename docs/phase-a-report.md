# Phase A: Agent tool contract and execution reliability

The implementation changes the execution protocol, not retrieval ranking or
model selection. The comparison freezes `ea6e395` as the baseline and `c47a015`
as the candidate. All later work is experiment execution, offline audit and
reporting.

## Architecture and compatibility

`Runtime.ask/run_agent` still composes live `DocumentAccess`/`ExactRetriever`
and one captured `RetrievalEngine` snapshot. A new `ToolSession` owns the
validated model-facing boundary and evidence registry for one run:

- Search and match expose `ref`, `source`, `title`, and verbatim `content`.
  The registry retains document ID, revision, chunk/section ID, coordinates,
  and indexed snapshot identity. Identical evidence reuses its reference within
  a run; a random run namespace prevents references from aliasing another run.
- The normal expansion is `read(ref=...)`. `expand="snippet"` rereads the bound
  excerpt; the default `document` expands the whole document. A known filename
  can be read with `read(source=...)`. Selectors cannot be combined. Internal
  range/section APIs remain available through the low-level document adapters.
- Resolution reads one current document, checks the bound source/ID/revision,
  and checks the original excerpt against its coordinates before returning
  content. It does not clamp invalid coordinates or substitute another source.
  A known filename can explicitly refresh content after a stale reference.
- Expected argument, pattern, reference and missing-source errors return
  `recoverable_error`. Unexpected backend/invariant errors end the run with
  `fatal_error`. `DocumentNotFound` distinguishes ordinary missing documents
  from internal `KeyError`/`IndexError` bugs. SDK argument-validation errors
  are recoverable only for the tool-arguments validation path.
- Every run records model requests/responses, tool attempts and results,
  reference mappings/resolutions, provider usage, evidence accounting, skipped
  or withheld work, errors, and its canonical final result.

The Agent uses `finish(answer, status, evidence_refs)` for an early answer.
The last of the existing eight model requests is reserved for constrained JSON
finalization with evidence tools disabled. Collection-budget exhaustion also
transitions to that phase. Rejected collection calls consume collection
allowances; a rejected finish proposal consumes a model turn and skips the
rest of that proposal batch. Finalization does not increase the total turn or
wall-clock allowance. An exhausted wall-clock deadline can still prevent an
answer.

`AgentFinal` is the canonical result: answer, status (`answered`, `partial`,
`insufficient_evidence`, or runtime `error`), resolved citations, termination
reason and error details. Its JSON serialization is deterministic. `response`
is the existing answer-text projection; `stop_reason="final"` indicates a
successfully accepted final, while the canonical termination reason also
records a budget-triggered finalization. Fatal runs retain a structured failure
and cause a nonzero CLI exit. Final proposals with invalid citations, duplicate
fields or non-schema prose are rejected; normal evaluation does not extract
JSON from Markdown or surrounding prose.

The separate generation package's claim/quote citation validation remains
independent. Agent reference validation establishes identity and serialization
correctness, not semantic entailment.

## Exact execution

Case-sensitive literal matching runs in process without spawning rg. Regex
and Unicode case-insensitive matching use one rg invocation over separately
materialized normalized documents. Separate files preserve document boundaries
and Rust-regex semantics; sorted internal names preserve source/position order.
The matcher keeps flat live-file scope, source filters, title/body normalization,
Unicode character coordinates, BOM behavior, filename matching and literal
limits. No BM25 index substitutes for live matching.

Each call has a default 30-second deadline, reduced to the run's remaining
allowance. Literal scans check cancellation/deadlines between documents and
matches; rg is polled and killed/reaped on timeout or cancellation. Timeouts
are explicit recoverable observations, not empty successful results. Local
file I/O remains synchronous, so its operating-system latency is not a hard
real-time guarantee.

## Original failures and fixes

| P4 failure | Implemented fix |
| --- | --- |
| Per-document rg spawning, including a run exceeding 17 minutes | In-process literal scan and one batched rg invocation; deadline/cancellation handling |
| Filename used as document ID, malformed IDs, mixed document/section/source, invalid offsets | Opaque run-scoped references and a simplified read schema |
| Deleted, renamed or edited sources | Explicit missing/stale-reference errors and validated source refresh |
| One bad tool call aborts the run | Narrow input/error classification, recoverable tool observations, charged attempts |
| Tool/evidence budget stops without an answer | Close collection and finalize from previously delivered evidence |
| Last tool turn consumes all model turns | Reserve the final request for constrained output |
| Markdown-wrapped or supplemented MuSiQue JSON fails parsing | Production finish/constrained schema and a direct benchmark adapter |
| Valid-looking but unknown citations | Resolve delivered references and recheck live revisions before accepting the final |
| Infrastructure exceptions lose the outcome | Canonical error state with stage/type/message and partial diagnostics |

No embedding model, Agent model, semantic algorithm, BM25 ranking, fusion/RRF
parameter, reranker model/allocation, chunking or indexing identity algorithm
was changed. A direct Pydantic dependency now declares the SDK validation type
already present in the lockfile; no dependency versions changed. Historical
P4 archives and datasets remain intact. The offline replay tool supports both
archived observations and the new canonical representation.

## Major files

| File | Responsibility |
| --- | --- |
| `src/arkb/agent/session.py` | Input validation, reference lifecycle, safe expansion, citation resolution |
| `src/arkb/agent/tools.py` | Existing capability adapters and new model-facing schemas |
| `src/arkb/agent/loop.py` | Recovery, collection/finalization lifecycle, terminal failure boundary |
| `src/arkb/agent/state.py` | Canonical final and detached diagnostic trace |
| `src/arkb/agent/observation.py` | Attempt, usage, evidence and phase accounting |
| `src/arkb/retrieval/exact.py` | Bounded live literal/rg matching |
| `src/arkb/knowledge/documents.py` | Explicit expected missing-document exception; underlying identity unchanged |
| `src/arkb/interfaces/cli.py` | Structured outcomes and error exit status |
| `src/arkb/evaluation/` | Canonical benchmark mapping, evidence-map replay, reliability accounting |
| `evaluation/experiments/run_phase_a_reliability.py` | Frozen source/case registration and old/new execution |
| `evaluation/audits/summarize_phase_a.py` | Independent source-span, accounting, request and citation audit |
| `tests/agent/`, `tests/retrieval/test_exact.py`, related runtime/evaluation tests | Contract, failure, freshness and regression coverage |

## Verification

- Normal deterministic suite: **1102 passed, 0 failed**, 59 service tests
  deselected. This includes **235 evaluation regression tests**; they are a
  subset of 1102, not an additional count.
- Service integration: **59 distinct tests passed**, including real Agent
  conversations, CLI entry points, persisted Qdrant/SQLite retrieval, model
  token accounting, citation generation and all four reranker checks.
- P4 real edit/delete/rename and publication fixture: **16 checks passed,
  0 failed**, including reference rejection and both remote snapshots matching
  SQLite. This is an experiment check count, separate from pytest cases.
- Architectural exact-match checks include 300 synthetic documents: zero
  subprocesses for ordinary literals, at most one for regex/case folding, plus
  real subprocess cancellation/reaping and source/Unicode isolation.

Integration setup attempts are retained: the first run passed 55 tests, failed
one uncached-default-path reranker test and skipped three unconfigured snapshot
checks. With the repository's existing `.arkb/models` cache, the frozen reranker
check passed; three checks then rejected the old documented storage-v1 snapshot.
Pointing them at the existing current-format example-notes snapshot yielded
three passes. No model weights were changed or downloaded to resolve this.

## Controlled comparison

The fixed set contains Stack Overflow IDs 7 and 9, Robotics IDs 4 and 7,
and the first two registered MuSiQue pairs (four variants). Each arm used
`qwen3.5:4b`, temperature 0, thinking enabled, eight model requests,
12/10/6 collection/query/read limits, 4000 evidence tokens, a 120-second
cooperative run deadline and the existing 180-second harness ceiling.

| Reliability measure | Baseline | Candidate |
| --- | ---: | ---: |
| Collection-call validation errors | 0/34 (0%) | 1/34 (2.94%) |
| Fatal tool errors | 1/34 (2.94%) | 0/34 (0%) |
| Recoverable tool errors | 0/34 (0%) | 1/34 (2.94%) |
| Runs producing final answers | 5/8 (62.5%) | 7/8 (87.5%) |
| MuSiQue strict structured predictions | 1/4 (25%) | 4/4 (100%) |
| Canonical envelopes serializable | Not available | 8/8 (100%), including one error |
| Agent exact-match errors/timeouts | 1/4 calls (25%) | Not defined: zero match calls |
| Mean requested/attempted collection calls | 4.25 / 4.25 | 4.25 / 4.25 |
| Mean finish calls | 0 | 0.50 |
| Mean model requests | 4.75 | 5.25 |
| Mean run elapsed time | 40.17 s | 18.04 s |

Baseline stops: five final, one fatal exact timeout, one evidence-budget stop,
and one turn-limit stop. Candidate termination reasons: four early finish,
one successful evidence-budget finalization, two successful reserved-turn
finalizations, and one invalid final output.

The candidate's single bad reference was reported as recoverable, and the run
continued to a final result. The remaining failure was Robotics ID 7: its last
schema-constrained request returned only reasoning and whitespace content,
with provider `done_reason=stop`. The runtime retained a structured failure;
it did not fabricate an answer. Thus **100% envelope serialization is not
100% final-answer production**.

Tool selection changed with the necessary contract instructions. In particular,
the candidate Agent made no match calls, so its Agent-run results alone cannot
measure the matcher improvement. A separate replay of the baseline's exact
same call, `match(query="pandas add column based on conditionals", target="content")`,
on the same 109,188-document Stack Overflow corpus completed in **19.89 s**,
returned zero matches and spawned **zero subprocesses**. The old tool attempt
was interrupted after **173.94 s** by the run's 180-second harness ceiling.
This is a single fixed-call diagnostic, not a throughput benchmark.

The offline audit verified **244 source spans, 80 model requests, and all seven
candidate final references**. It independently recomputed evidence token totals,
provider-usage totals, conversation/request correspondence and final proposals.
Snapshot identities and SQLite file hashes stayed unchanged, and recorded
model digests, templates and service/software environments agreed between arms.

Answer/retrieval diagnostics are retained without tuning:

| Diagnostic | Baseline | Candidate |
| --- | ---: | ---: |
| MuSiQue answer F1 | 0.5000 | 0.0786 |
| MuSiQue answer exact match | 0.5000 | 0.0000 |
| MuSiQue support F1 | 0.5000 | 1.0000 |
| MuSiQue answerability accuracy | 0.2500 | 1.0000 |
| Paired answer-sufficiency F1 | 0.0000 | 0.0786 |
| Paired support-sufficiency F1 | 0.0000 | 1.0000 |
| Submitted positive-document recall, macro | 0.4643 | 0.5714 |
| Submitted weighted-aspect coverage, macro | 0.1250 | 0.3214 |

The MuSiQue metrics cover only two pairs; recall/coverage averages exclude
undefined gold denominators (the machine-readable summary records their
counts). **Raw answer F1 decreased.** The old benchmark suffix requested short
answers; the new canonical answers include explanations, which token-overlap
F1 penalizes. Both answerable candidate outputs contain the gold target names,
but their additional factual claims have not received independent review.
This content-format confound was declared before execution.
No answer shortening, gold-based extraction, prompt retuning or repeated
score selection was applied. These numbers do not establish improved answer
correctness. Bright answers still need independent semantic review.

A preparation-only harness amendment is retained: the first attempt used the
wrong MuSiQue context name when reconstructing source filenames. File-set
validation rejected it before any MuSiQue Agent call. The four completed
Bright baseline rows were preserved, and the run resumed only the four
unexecuted MuSiQue variants after correcting that name. There is still exactly
one model trial per registered case per arm; measured production sources,
models, prompts, budgets and case selection were unchanged.

Artifacts:

- [Machine-readable comparison](../evaluation/phase-a/reliability-v1/summary.json)
- [Frozen protocol](../evaluation/phase-a/reliability-v1/protocol.json),
  [registered cases](../evaluation/phase-a/reliability-v1/cases.json),
  [harness amendment](../evaluation/phase-a/reliability-v1/harness-amendment.json)
- [Complete run archive](../evaluation/phase-a/reliability-v1/run.tar.gz) and
  [147-file SHA-256 manifest](../evaluation/phase-a/reliability-v1/archive-manifest.json).
  The archive contains both source snapshots, raw model/tool traces, tokenizers,
  gold for the four MuSiQue variants, environments, runner and failed/resumed
  harness logs. Original large P4 corpora/indexes remain in their existing locations.
- [Verification counts](../evaluation/phase-a/verification/test-summary.json) and
  [live freshness evidence](../evaluation/phase-a/verification/freshness.json).
  JUnit and setup-attempt logs are retained beside these files.

To replay the offline audit, extract the archive to a new directory and run
`python evaluation/audits/summarize_phase_a.py <extracted-directory>` with the
existing registered P4 corpora available. A new live run must use a new output
directory with `run_phase_a_reliability.py --register`, then execute each arm;
never overwrite an earlier trial.


## Remaining issues

Tool/runtime: the observed empty finalization response remains a model/provider protocol limitation. Constrained generation can still propose an unknown/stale
citation or ignore its schema; the runtime reports failure if its remaining
allowance cannot repair it. Provider requests retain transport timeouts and a
cooperative run deadline. The SDK can reject malformed arguments before
returning the full assistant response: diagnostics retain its validation
locations and offending arguments, but cannot recover unreturned sibling
calls. Evidence counts remain an explicitly labeled reference-token ruler,
not a guarantee of the provider's rendered context window.

Agent policy: stopping efficiently, choosing useful expansions, avoiding
repeated searches, deciding whether evidence supports an answer, and writing
complete concise answers remain model decisions. Citation identity validation
does not establish factual support. A no-tool answer to a knowledge-dependent
question cannot be ruled out by serialization checks alone.

Retrieval quality: missing relevant documents, incomplete multi-hop evidence,
and distractor ranking require separate evaluation and subsequent work.
This phase implements no reranking, retrieval-policy training or benchmark
heuristics. Small-sample protocol results do not replace independent answer
review or establish release readiness.

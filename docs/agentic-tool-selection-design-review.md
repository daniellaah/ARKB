# Agentic evaluation design review

Date: 2026-09-14 UTC. Status: current core execution stopped for design review;
replacement design proposed, not executed or registered.

Update: the [tool overhead repair](tool-overhead-repair.md) is implemented and
verified by full-corpus development replay: 87 match calls passed, including all
25 historical timeouts; BM25 probe responses remain identical. This completes
the profiling/repair work below, not the replacement protocol or its dispatch
gates. The old core and heartbeat remain paused.

## Decision and preserved evidence

The user challenged the approximately 50-hour run after inspection revealed
systematic text-match timeouts. Suspend `core-v1` and the `arkb-agentic` heartbeat.
Do not resume, grade it as a completed core study, or combine it with a replacement
experiment. Its frozen protocol and measured source remain unchanged.

The stop preserved 11 completed attempts and one administratively interrupted
attempt. The remaining 5,448 scheduled attempts were not started. Neither the
interrupted attempt nor unstarted work is a model failure. The worker's native
`failed`/`KeyboardInterrupt` status describes the process interruption; the
separate administrative record explains why it happened. Retain partial provider
logs without constructing a synthetic final response or silently retrying them.

The previous two 98-attempt pilots, calibration material, full Phase C corpora,
indexes, embeddings and reports remain available. Phase C is still completed.

Operational evidence is in:

- `evaluation/agentic-tools/design-review-v1/execution-cost-audit.json`;
- `/Volumes/ARKBPhaseC/agentic-tools-v1/core-v1/administrative-stop-request.json`;
- `/Volumes/ARKBPhaseC/agentic-tools-v1/core-v1/administrative-stop-accounting.json`;
- `evaluation/results/agentic-tools-v1/orchestration-status.json`.

## What the evidence establishes

The replacement pilot's BrowseComp A-M arm made 28 match calls across four
trajectories; 23 returned `exact_timeout`. Mean trajectory time was 208.86 seconds,
including 202.11 seconds in matching and 6.74 seconds in model calls. The first
completed core A-M trajectory also spent approximately 211 seconds in seven
match calls, all of which timed out.

The pilot-based projection apportions the planned 49.67 inference hours to about
20.69 hours of matching, 9.60 hours of other search and 18.39 hours of model calls,
with remaining time in reads, finalization and harness overhead. These are small
pilot extrapolations. Tool timing is not a direct GPU utilization measurement,
and search timing does not identify the cost of each internal retrieval leg.

`DocumentAccess.records()` reads and parses live files on each call. Literal
matching scans those records in Python. Regex and case-insensitive matching also
materialize normalized documents into temporary files before invoking ripgrep.
This is an implementation bottleneck to profile and fix. These measurements do
not establish that ripgrep itself is slow. BrowseComp BM25 and Hybrid also need
separate profiling: their pilot tool time is material and is not explained by
the match implementation alone.

These observations support a diagnosis of the current product's operational
limitations. They do not isolate the benefit of autonomous tool selection.
Timeouts remain real outcomes under the old version; this review does not erase
them, recategorize them all as infrastructure faults, or establish a winning arm.

## Design and dispatch mistakes

1. The technical review emphasized final-output validity, dependencies and trace
   integrity. Recoverable tool errors were retained but did not block dispatch,
   so systematic match timeouts could coexist with a technical pass. Integrity
   checks establish faithful recording, not suitability for a long experiment.
2. The study combined operational reliability, answer quality, tool-choice
   behavior and isolated latency into one expensive schedule. A severely limited
   tool implementation obscures attribution to the Agent's selection policy.
3. Three repetitions for every arm and dataset were fixed without an empirical
   variance or precision justification. They improve repeatability measurement
   but do not create three times as many independent questions. Temperature zero
   alone does not establish deterministic outputs.
4. Serial execution applied to every quality trial to protect timing comparisons.
   This couples total throughput to CPU/file delays. Concurrency must be evaluated
   explicitly because queueing and shared-resource contention can change deadline
   failures as well as latency.
5. The original plan allowed fixed-RAG retrieval caching, but the executed path
   repeated retrieval each time. Any future caching must preserve the same ranked
   evidence and report its preparation/retrieval cost separately.
6. Automatic calibration supplied an execution check with only four eligible
   answer judgments. It did not establish grader accuracy. Human review remains
   pending; no definitive human-verified quality claim is available.

The seven capability arms, common model and evidence rules, full corpus,
query-level pairing, preservation of failures and separated scoring labels are
still useful controls. They do not need to be discarded because dispatch failed.

## Replacement design, to freeze before new benchmark inference

### 1. Separate the questions

Report current-version product reliability from the preserved diagnostics.
Evaluate tool-selection benefit on one new, shared implementation only after
tool execution is fit for the intended workload. Keep all seven original arms
in the registered comparison; do not omit a weak arm based on observed answers.
Use a separate, preregistered latency subset to measure isolated cold/warm costs
and a separate run to measure system throughput under concurrency.

### 2. Establish tool readiness on the full corpus

Profile document enumeration, decoding/normalization, temporary writes, rg
execution, result mapping and BM25/Hybrid substeps independently. Prepare reusable
normalized content if profiling supports it; bind it to source revisions and
preserve live-file freshness, span coordinates, ordering, filtering, cancellation
and timeout behavior. Validate output equivalence on known-hit, absent-hit,
regex, case-folding and changed-file fixtures. Do not replace the full corpus
with only relevant documents or silently shrink the benchmark's distractor set.

Implement a readiness gate that reads tool events, not only final statuses.
Before running it, freeze a representative query workload, full-corpus coverage,
latency targets and failure thresholds. A proposed engineering target is zero
unexpected failures on valid known-result contract fixtures, at least 99% timely
completion on a fixed performance workload, and p95 tool time below half of the
30-second call ceiling. These are proposed acceptance targets, not measured
results, universal benchmark rules or thresholds to loosen after a failed run.
Expected errors from deliberately invalid requests must be reported separately.

After those checks, rerun the complete pilot using the same corrected source for
all arms. Publish failures by dataset, arm, tool and error type. A repeated valid
call pattern that exhausts its deadline must trigger review before expansion.
Low answer accuracy alone must not be a readiness failure: never tune, exclude
or retry arms merely to obtain better scores. Freeze the final pilot stop rule
before its execution, including how Agent-generated invalid calls are classified.

### 3. Plan statistical value and cost together

Choose the target effect and interval precision for the four primary contrasts,
then justify independent query counts and repetitions with prespecified paired
variance scenarios and a small repeated execution study. Do not automatically
retain or drop the three-repetition requirement. Publish compute and storage
budgets, measured component costs and the decision before core dispatch. Do not
use observed core quality to increase the sample until significance appears.

A one-repetition pass through the old manifest would contain 1,820 trajectories,
but that arithmetic alone is not a replacement protocol or a statistically
justified sample size. Additional repetitions, dataset allocation and fixed-RAG
retrieval reuse need explicit decisions before inference.

### 4. Validate the execution mode

Benchmark concurrency levels one, two and four on an isolated development
workload after tool repair. Measure throughput, per-request service/queue time,
memory, errors, effective model settings and deadline effects. Separate
preparation from measured work. Adopt concurrency only with documented resource
and outcome behavior; do not promise linear speedup or describe concurrent
latency as isolated latency. If concurrency affects quality through deadlines,
report the serving policy as part of the tested system or use serial quality
trials. Keep the isolated latency subset independently randomized and balanced.

### 5. Freeze a new version and protect interpretation

The replacement requires a new protocol, source snapshot, pilot records and
attempt namespace. Keep the old pilot/core diagnostics separate. Treat core IDs
whose trajectories were inspected as exposed development material; select fresh
confirmation IDs by an outcome-independent, recorded reserve rule if needed.
Do not claim a new holdout merely because a directory or protocol hash changed.
Declare which original datasets or labels were already public or inspected.

Keep all trials in the relevant denominator, group repetitions within questions,
preserve MuSiQue pairs, register comparisons and uncertainty methods, and retain
separate evidence and answer metrics. Complete or explicitly qualify grader and
citation review. Add a graceful between-attempt stop mechanism and administrative
accounting so a future design hold does not require interrupting an active trial.

## Resume conditions and remaining work

No replacement run is authorized by an old `core_dispatch_authorized` flag alone.
The user has already authorized work on evaluation optimization; implementation
and development diagnostics can proceed within that scope. Formal execution
must satisfy the revised, recorded gates first. The paused heartbeat must not
restart the old schedule while those gates are unresolved.

Pending work: detailed tool profiling and correction; executable readiness and
graceful-stop checks; measured concurrency and variance studies; a justified,
versioned replacement protocol and full pilot; then new core execution, grading,
independent review qualification, audits and reporting. This design review does
not claim those steps are complete or promise a new completion time.

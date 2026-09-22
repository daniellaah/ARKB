# Phase A implementation plan

Baseline: `ea6e395` (clean working tree). This phase changes the Agent protocol,
not models, indexed retrieval, ranking, chunking, or benchmark policy.

## Audited path

`Runtime.ask/run_agent` composes `AgentTools` over live `DocumentAccess` and
`ExactRetriever`, plus a captured `RetrievalEngine` snapshot. The Ollama loop
advertises match/search/read and directly expands model arguments into Python
calls. Search results carry internal document/revision/chunk/section identities
and body coordinates. Read accepts those independently; expected input and lookup
errors currently escape the loop. Optional `AgentObserver` records requests,
results, tokens and budget stops. Tool/evidence stops and the last tool turn can
discard the final-answer opportunity. Final responses are arbitrary text;
MuSiQue's adapter parses a separately prompted JSON object. Exact matching runs
rg for every normalized document. CLI and evaluation consumers use response,
stop_reason, messages, trace, and optional observation.

## Implementation sequence and public test seams

1. Exact execution: in-process case-sensitive literal scan; one bounded rg call
   over normalized temporary documents for Rust-regex and Unicode case folding.
   Preserve body offsets, source ordering, flat live scope, errors and early
   literal limits. Test ExactRetriever with real files, including subprocess
   count, cancellation and timeout.
2. Run-owned tool session: opaque references bind full internal evidence;
   schemas expose `read(ref, expand)` or a known source filename. Validate model
   arguments before execution. Expected input/reference errors are recoverable;
   infrastructure/invariant failures are fatal. Keep low-level DocumentAccess
   range reads and diagnostic evidence identities. Test actual files and engine
   outputs, including edits, deletes, renames, mixed selectors and revisions.
3. Lifecycle/output: add a finish tool and canonical Agent final result, reserve
   the last model turn for schema-constrained finalization, and transition tool
   or evidence exhaustion to that phase. Count invalid attempts. Record every
   request, error, reference resolution, withheld result and canonical outcome.
   Test scripted model sequences through run_agent and the real SDK transport.
4. Adapt CLI/evaluation at their existing boundaries. New benchmark runs map
   canonical results directly; historical text parsing remains for old archives.
   Run focused checks, the complete normal suite, and available service-backed
   indexing/freshness/Agent integration checks.
5. Freeze code before one controlled reliability comparison. Use the first two
   registered Stack Overflow cases, first two Robotics cases, and first two
   MuSiQue question pairs (four variants), in registration order: eight cases
   per implementation, one trial. Reuse each exact P4 corpus and index snapshot.
   Old source is ea6e395; new source is the Phase A implementation. Same 4B model,
   temperature 0, think=true, eight total model requests, 12/10/6 tool/query/read
   limits, 4000 evidence tokens, 120-second runtime and 180-second harness limit.
   Only necessary tool/output-contract instructions differ (old MuSiQue output
   suffix versus canonical adapter). Record this confound; do not tune on scores.
   Report validation/recoverable/fatal errors, output and schema validity, exact
   errors/timeouts, calls/turns, termination, evidence coverage and MuSiQue scores.

The user's Task 10 supplies the test-boundary and behavior requirements; proceed
with red/green vertical slices. Commit logical units and finish with a clean
tree. Preserve historical reports/data and distinguish protocol validity from
answer correctness and evidence sufficiency.

# Repaired tool pilot v3

Status: registered when the new executable protocol and source snapshot are
written; this amendment does not authorize a replacement core schedule.

## Purpose and retained controls

Rerun all 98 development trajectories (14 scenarios, seven arms, one repetition)
from pilot v2 using the repaired production tools. Keep the same complete corpora,
indexes, selected questions, prompts, model identities, nonthinking mode, context,
output and evidence budgets, candidate depth, reranker setting and arm order.
These exposed pilot questions are development material, not a new holdout.
Do not combine v3 results with old pilots or the interrupted core-v1.

The repair evidence is the complete 87-call match replay, 252 differential checks,
1,251 regression tests and identical BM25 probe response hashes, archived under
evaluation/agentic-tools/tool-overhead-v1. The new runner prepares disposable exact
text once per active scope and records setup separately from query time. It closes
the prior scope's cache before preparing the next one. No corpus or vector rebuild.

## Frozen operational gate

readiness-policy.json is frozen with the executable source before any v3 trial.
For each corpus/context, ten fixed synthetic calls exercise absent/common literal
and regex matching, three BM25 queries, semantic, Hybrid and a deterministic source
read. Every call uses the entire registered scope. Check live text/revisions for
returned evidence; the absent matching probe must return no hits. The first source
in deterministic DocumentAccess order is used for the read fixture. These checks
run after preparation and before the first measured trajectory of that scope.
Their time is recorded separately. All scopes must pass for final pilot readiness.

Both scope fixtures and executed pilot tool calls require zero unexpected errors,
at least 99% completion below 30 seconds, and nearest-rank p95 below 15 seconds for
each dataset/effective-tool group. Also report results by dataset/arm/tool. These
are engineering acceptance rules on a finite development workload, not confidence
bounds on population reliability. Small groups do not establish a 99% service SLA.

Expected boundary rejections (invalid arguments, pattern, references/citations or
unknown tool) are reported separately and retained in the trajectory denominator.
A read of a nonexistent model-supplied filename is an input error; disappearance
of a bound reference or a stale revision is an operational failure on this frozen
corpus. Unknown/fatal failures default to operational failure. Skipped calls are
reported by the trace and are not fabricated as executions. finish calls also
undergo the tool gate because they validate live references.

After any unexpected tool failure, valid call reaching 30 seconds, failed provider
request, missing trace, harness failure or detected competing GPU process, finish
and retain the current trajectory, then pause before another attempt. Preserve
the cause in stop-request.json; restarting the worker cannot bypass that file or
an operational failure in a completed attempt. No automatic retries or threshold
changes are permitted. Final p95 is checked after the complete pilot. Low answer
accuracy, insufficient evidence, token exhaustion and invalid model final output
are measured outcomes; they never justify dropping an arm or tuning on answers.

SIGINT/SIGTERM and a stop-request.json file request an administrative pause. The
worker completes its current attempt under the existing 360-second harness guard,
writes result/provider checksums, and pauses between attempts. Setup can take time
to finish before a pause. Forced process termination may still require explicit
interruption accounting. No promise of immediate SIGTERM cancellation is made.

## Execution and subsequent decisions

Use a new pilot-v3 output directory and protocol-derived attempt IDs. Inherit the
original root inference.lock even though the parent lives in pilot-v2. Verify
actual provider options and journals, immutable inputs, source snapshot and model
identities. GPU competition is checked live before scope checks and each attempt.
The v3 runner must execute from its frozen source, not the working checkout.

This pilot uses concurrency one to obtain an uncontended repaired baseline. That
is not a decision to serialize the eventual core. After the pilot, register a
separate development workload for concurrency 1/2/4 and repeated execution; measure
throughput, queue/service time, effective settings, errors and output variability.
Use these results and prespecified precision/effect scenarios to choose core query
counts and repetitions. Register those choices before core inference. The copied
5,460-attempt schedule is only a historical capacity scenario, not dispatch approval
or a newly justified study. No new total-duration promise follows from old timings.

Human grader/citation review remains pending. Automatic calibration can support
execution checks but cannot establish human agreement or definitive quality claims.

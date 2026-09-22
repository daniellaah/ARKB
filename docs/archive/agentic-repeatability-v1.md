# Full Agent repeatability study v1

Register before the additional trajectories. Serving-v1 completed 90 measured
calls with no operational errors and less than 3% median throughput improvement
from client concurrency 2 or 4. Neither passed the frozen 10% screening rule.
The installed Ollama 0.33.2 scheduler constrains the qwen35 family to one execution
slot; the live llama-server command confirmed `-np 1`. This study therefore keeps
one active trajectory and the same serving implementation/model as pilot v3.
Backend migrations or isolated replicas would require separate validation.

## Workload and retained baseline

Select the first v3 pilot unit in schedule order from each dataset, retaining
both MuSiQue variants. Keep every arm for these five scenarios: 35 baseline
trajectories already measured in v3. Add exactly two new repetitions per
scenario/arm, 70 new trajectories, for three executions per cell. Selection uses
only schedule identity, not answers, correctness, failure, latency or the serving
probe's generated content. These are four independent questions, including one
MuSiQue pair, not 105 independent observations or an unbiased corpus-wide sample.

Use v3 repetition zero as an explicitly historical development baseline. The
new repetitions are one and two. Rotate arm order by (original unit_index +
repetition) modulo seven, with MuSiQue variants adjacent. Finish all repetitions
of the selected unit before moving to the next dataset; reuse each prepared
full-corpus engine. Full corpora/indexes, model identities, prompts, tool sets,
budgets, exact preparation, reference rules and transport remain the same. New
sessions keep normal opaque evidence IDs. Corpus/model caches and elapsed time
between v3 and this run can affect latency; do not treat their raw latency
difference as a randomized serving-performance comparison.

Freeze new source and protocol identities. Require all production source,
capability/transport/budget code, dependency checks, schedule-key function,
readiness and stop-controller files to match v3. The runner change is restricted
to unique auxiliary fixture filenames; accounting accepts the registered count
instead of a hardcoded 98. Each repeated scope check now keeps its own artifact.
Do not rewrite or repair the original v3 records.

## Checks and reporting

Retain the exact v3 operational thresholds and stop policy. All 70 new attempts
and their provider journals must be intact; all required scope checks must pass.
Keep errors and nonanswers. A tool/provider failure pauses before another attempt;
diagnose it before any explicit resume. There are no automatic retries or output
repairs. Original root inference.lock and live GPU-competition checks apply.

Compare complete normalized final objects, delivered/submitted/submitted-citation
evidence identities, executed tool names/arguments, model-request counts, stop
reasons and elapsed times across each cell's three executions. Replace known
opaque evidence references with stable content/revision/source/span identities
only for offline comparison; never alter model inputs or generated outputs.
An exact text difference is an output-stability observation, not a correctness
judgment. Unknown/invalid references remain visible. Report comparisons within
the two new repetitions and separately against the historical baseline.

This small workload detects repeated execution differences and measures current
full-Agent workload costs. It cannot establish a population intraclass correlation,
prove determinism, set a 99% reliability SLA or alone justify the core's repetition
count. Do not expand it in response to favorable or unfavorable answers. A fixed
105-execution descriptive analysis completes this study; no core dispatch follows
automatically.

Before the replacement core, publish paired effect/precision scenarios (including
finite available questions and repeated-run correlation sensitivity), query and
repetition counts, compute budget and uncertainty methods. Keep the original four
primary contrasts and MuSiQue pairing unless a new amendment explains a change
before inference. Grader/citation review remains separately qualified.

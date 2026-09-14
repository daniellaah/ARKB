# Agentic core execution protocol

The replacement pilot completed all 98 attempts at 2026-09-14T03:29:30Z. Its
1,184 trace checks passed. The missing-ripgrep and empty-final defects did not
recur. One response exhausted the generation ceiling and two finals contained
invalid references; all three remain model failures in the pilot denominator.
No answer-correctness scores were used to change retrieval, selection or budgets.

The core run retains the [v2 amendment](agentic-tool-selection-pilot-amendment-v2.md):
the same exact Qwen3.5-4B weights, nonthinking mode across every request and arm,
32,768 context, 4,096 output, 8,000 evidence-body tokens, eight model turns,
12/10/6 tool/query/read ceilings, 300-second cooperative and 360-second harness
deadlines. Capability definitions, Agent loop, fixed-RAG packing and all
production source remain identical to the successful v2 pilot. There is no
reranking, corpus reduction, cache-based answer reuse or new retrieval tuning.

## Work and cost

Execute the already frozen schedule: 100 BrowseComp queries, 50 FiQA queries,
50 NFCorpus queries and 30 MuSiQue pairs (60 original variants), seven arms and
three repetitions, totaling **5,460 trajectories**. Preserve the registered
cyclic arm rotation and adjacent paired variants. Fixed RAG repeats retrieval
and generation; its measured pilot cost therefore describes the actual core path.

The completed pilot projects **49.67 hours of sequential core inference** and
**1.382 GiB** of raw attempt artifacts. Preparation took 365.5 seconds in the
pilot and is recorded separately. These estimates come from a small pilot and
are not a promised completion time. Judge execution, interruption/reload costs,
independent review and final archival are additional. The external APFS volume
had approximately 70 GiB free during preparation, sufficient for the projected
new traces without replacing preserved artifacts.

The four eligible calibration judgments averaged 8.31 seconds each. If all
2,100 core BrowseComp responses require judging, that would add about 4.85 hours
before retries, warmup and review. The eligible count is unknown before core
execution; the four-response timing sample provides only a rough planning value.

## Scoring and review

BrowseComp answer success uses every attempted trajectory. A nonblank canonical
`answered` or `partial` final is eligible for the frozen official rubric. Errors,
absent finals and `insufficient_evidence` are zero; unresolved grader failures
are pending, not model errors. The unchanged request may be retried twice only
for grader transport, truncation or parse failure. Incorrect judgments are not
retried. The frozen parser, prompt, local Qwen3-32B Q8_0 identity and options are
an explicitly disclosed ARKB adaptation of the official setup.

Calibration exported all 42 prescribed responses, passed synthetic positive and
negative controls, and completed automatic labels without unresolved parsing.
Only four calibration responses required the answer judge; other responses
were canonical nonanswers/errors or unanswerable MuSiQue variants. Consequently,
this is a limited execution check, not proof of grader accuracy. MuSiQue
unanswerable calibration cases use canonical answerability rather than a judge
guessing an unavailable fact. Independent human labels and agreement remain
pending. All LLM answer-quality results are **provisional** until reviewed.

MuSiQue primary metrics use the existing strict canonical mapping and normalized
answer EM/F1, support F1, answerability and pair sufficiency. Partial maps to
answerable=false; missing/error finals cannot earn abstention credit. Support
identities come only from the frozen scoring map. Answer metrics are computed on
the answerable member and pair sufficiency additionally requires both variants'
answerability decisions to be correct. Repetitions are averaged within the same
original pair. FiQA/NFCorpus provide evidence diagnostics, not answer correctness.

Export the first 20 registered core BrowseComp queries, repetition zero and all
seven arms as 140 blinded citation-review responses. Include exact cited excerpts
and source/revision/span provenance; keep arm mappings separate. Human claim
labels remain null until supplied. Empty or abstained responses are separate
from unsupported claims. Templates and completed review submissions are separate
artifacts; do not overwrite frozen review inputs or manufacture human agreement.

## Analysis and integrity

Keep returned, delivered and model-submitted evidence separate. Unjudged sources
remain unknown, and accumulated Agent evidence does not acquire an invented
nDCG ranking. Completed internal retrieval legs are derived from successful
calls to the frozen sequential engine: Hybrid contributes one BM25 and one
Semantic leg. A failed search may contain unknown partial work; explicitly
report that uncertainty rather than assigning zero execution cost.

Use 20,000 paired bootstrap resamples, seed 20260912, after averaging repetitions
within a query. MuSiQue resamples whole pairs within each hop stratum. All arms
share each resample. The four primary A-All versus restricted-Agent contrasts
use nominal 95% and Bonferroni-adjusted 98.75% intervals (0.625/99.375 percentiles).
A meaningful gain requires at least five percentage points and an adjusted lower
bound above zero. All four must meet the rule to claim superiority over every
restricted arm. Other comparisons are exploratory; no cross-dataset pooled score.
Pending judge labels block definitive answer comparisons rather than disappearing
from them. Report observed cost alongside quality and uncertainty.

Core-specific changes are scheduling and audit bookkeeping: the registered core
phase/count gate, full result/provider verification on resume, per-invocation
status history, cumulative reload/setup records, and a post-attempt check for
competing GPU work. Inference adapters and production code are hash-compared
with v2 before freezing. No measurement change silently modifies model inputs.

One shared inference lock prevents pilot, core and judge overlap. A completed
attempt is immutable and skipped only after identity/checksum verification.
An incomplete attempt requires explicit recovery accounting; do not silently
resubmit it. Keep all original and replacement attempts. A competing user task
causes the worker to yield between attempts; retain timing-contamination flags
and exclude such timings from clean latency comparisons while keeping their
quality outcomes. The checks are at boundaries, not continuous GPU monitoring.

After core and scoring, verify source/model/data identity, replay all traces and
metrics, preserve checksums and produce an English report with a Chinese summary.
Human review remains an explicit qualification, never a claimed completion.

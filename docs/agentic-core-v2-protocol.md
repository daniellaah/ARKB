# Replacement core v2 protocol

Status: prepared 2026-09-18 UTC from frozen, audited inputs; registered only when
the core-v2 pipeline writes `protocol.json` after resources and identities verify.
This document is copied into the run directory as `core-protocol.md`. It replaces
the suspended 5,460-attempt core-v1 schedule, which stays preserved as incomplete
historical diagnostics and is never combined with this run.

## Inheritance and gates

Every inference setting is the pilot-v3 setting: the same seven arms, Qwen3.5-4B
weights, nonthinking mode, temperature 0, 32,768 context, 4,096 output tokens,
8,000 evidence-body tokens, eight turns, 12/10/6 tool/query/read ceilings, 300 s
cooperative and 360 s harness deadlines, candidate depth 20, reranking off, the
full corpora and READY indexes, the same prompts and the repaired exact-match
cache. Production source and the capability, transport, dependency, selection,
readiness and stop-controller modules are hash-compared with the v3 snapshot at
preparation and again at registration. Nothing was tuned on answers.

Dispatch requires all of: the audited v3 pilot with zero operational tool
failures and a passed readiness gate; the negative serving-v1 concurrency screen;
the audited repeatability-v1 study; automatic calibration with passed synthetic
controls and scoring files identical to calibration; the outcome-blind design
reproduced from frozen inputs; a clean regression run; provider preflight; model,
server and ripgrep identities equal to v3; the shared inference lock free; and no
competing GPU work. Human grader calibration and citation review remain pending,
so every answer-quality result stays provisional.

## Questions and exposure

| Track | Registered core | Eligible after exclusion | Excluded development IDs |
| --- | ---: | ---: | --- |
| BrowseComp-Plus | 320 queries | 825 of 830 | 4 pilot queries; old-core question 555 |
| FiQA | 50 queries | 646 of 648 | 2 pilot queries |
| NFCorpus | 50 queries | 321 of 323 | 2 pilot queries |
| MuSiQue | 30 pairs (10 per hop stratum), 60 variants | 97 of 100 pairs | 3 pilot pairs |

Selection keeps the frozen hash order `sha256(agentic-tools-v1|20260912|dataset|
stratum|id)` and takes the first eligible IDs after removing pilot IDs and every
question the interrupted core-v1 started. FiQA, NFCorpus and MuSiQue therefore
retain the original core IDs; BrowseComp keeps 99 original core IDs and adds 221
reserve queries that no trajectory has ever used. Fresh BrowseComp scenarios take
only the pinned official query text; MuSiQue contexts and indexes are reused
unchanged. All labels are public benchmark labels; no private test set is claimed.
No core-v2 question was inspected before inference.

## Repetitions and estimands

Repetition zero of every registered question is the **primary run**: 3,360
trajectories. Its estimand is the mean single-session outcome of each arm over the
registered question distribution, including session variability, and the paired
A-All minus restricted-arm differences on the same questions.

The **repeat subset** is the first 20 registered BrowseComp IDs in hash order, the
same prefix as the citation-review sample, run in all seven arms for repetitions
one and two: 280 additional trajectories scheduled adjacent to repetition zero
with the usual arm rotation. Its estimand is between-session variability under the
identical protocol: a one-way random-effects decomposition of answer success and
of the four primary paired differences into between-question and within-question
components. Repetitions one and two never enter primary contrasts, evidence
comparisons, cost ratios or the citation review; there is no best-of-k.

Total registered trajectories: **3,640**.

Precision, not power at five points, is the registered target. With four primary
contrasts at family alpha 0.05 (per-comparison 0.0125), a half-width of 0.10 on a
paired success difference whose discordance rate is at most one half needs 312
questions; 320 are registered. For any repeat correlation rho, one repetition of n
questions has effective size n, whereas 100 questions with three repetitions have
100 / (rho + (1 - rho) / 3): at most 300 and only 100 when sessions are perfectly
correlated. Under discordance 0.25 the expected half-width is 0.069 (320 x 1)
versus 0.072 to 0.124 (100 x 3); under discordance 0.5 it is 0.098 versus 0.102 to
0.176. Normal-approximation power for a true five-point gain is 0.24 and 0.11 in
those two cases, so a null result is inconclusive rather than evidence of
equality. The decision rule is unchanged: a meaningful gain over a restricted arm
needs at least five observed points and a Bonferroni-adjusted 98.75% lower bound
above zero; superiority over every restricted arm needs all four. The full
scenario table is `precision-cost.json`.

## Execution policy

- One active trajectory. The installed Ollama scheduler gives qwen35 one execution
  slot and client concurrency gained under 3%; no serving change is made.
- Dataset order BrowseComp-Plus, FiQA, NFCorpus, MuSiQue; cyclic arm rotation by
  (unit index + repetition) modulo seven; MuSiQue variants adjacent.
- Before the first measured attempt of every scope, the frozen v3 ten-call
  readiness workload runs on the complete scope; a failed fixture pauses the run.
- Timeouts, tool errors, budget stops, invalid finals and nonanswers are measured
  outcomes retained in every denominator. They are never rerun, replaced or
  excluded, and answer quality never affects scheduling.
- The run pauses between attempts, after preserving the current attempt, when: a
  harness or provider-request failure leaves no trace; a competing GPU process is
  observed during an attempt; one attempt reaches three operational tool failures;
  the operational-failure rate among valid executed calls exceeds 1% once at least
  200 valid calls are in the current window; or measured trajectory time reaches
  the 32-hour cap. Pauses persist in `stop-request.json` and are diagnosed, never
  automatically retried. `operational-review.json` records reviewed attempt keys;
  reviewed attempts stay in the data, leave the pause rules and restart the rate
  window. The cap can be extended only by a recorded `time-cap-extension.json`.
- GPU competition is checked live before scope fixtures and every attempt using
  the named training/evaluation patterns plus any foreign Python process holding
  an Apple Metal device. The worker yields; it never terminates another task.
- Fixed-RAG retrieval is repeated on every attempt, as in the pilot that priced it.
- Resume verifies all identities and skips immutable completed attempts; an
  interrupted attempt requires explicit accounting. SIGINT/SIGTERM and the stop
  file request a graceful pause.

Harness changes relative to v3 are limited to the competitor detection, the
registered pause rules and the time cap. Offline analysis reads attempt and grade
counts and repetition roles from the protocol instead of hardcoded values.

## Cost

Applying v3 pilot means by dataset to the schedule gives 24.13 hours of measured
trajectory time: BrowseComp 20.58 h, FiQA 0.88 h, NFCorpus 1.47 h, MuSiQue 1.20 h.
With 25% headroom the planning figure is 30.16 hours; the registered cap is 32
hours of measured trajectory time. Excluded: exact-text scope preparation (about
ten minutes for BrowseComp on each worker start), scope fixtures, judge inference
(at most 2,520 BrowseComp responses; only nonblank answered/partial finals are
judged, about 8.3 s each in calibration), human review, administrative waiting
and archival. Expected raw attempt artifacts are about 1 GiB; the volume had 70
GiB free at preparation. These are small-pilot extrapolations, not a promise.

## Scoring, review and analysis

BrowseComp answer success is scored over all attempted trajectories with the
frozen official rubric under the disclosed Qwen3-32B Q8_0 Ollama adaptation;
errors, absent finals and `insufficient_evidence` score zero and grader failures
stay pending. MuSiQue uses the strict canonical mapping, answer EM/F1, support F1,
answerability and pair sufficiency over whole pairs within hop strata. FiQA and
NFCorpus report evidence diagnostics only. Returned, delivered and submitted
evidence stay separate; no Agent nDCG is manufactured.

Repetition zero of the first 20 registered BrowseComp IDs in all seven arms is
exported as 140 blinded citation-review responses with excerpt provenance. Paired
query bootstrap uses 20,000 resamples and seed 20260912 on repetition zero, with
nominal 95% and adjusted 98.75% intervals for the four primary contrasts, wins/
ties/losses and raw sizes. The repeat subset is reported separately as variance
components. Token and latency ratios accompany quality; timings with a competing
process flag are excluded from isolated-latency comparisons only.

## Artifacts

Output `/Volumes/ARKBPhaseC/agentic-tools-v1/core-v2`; public copies under
`evaluation/agentic-tools/core-v2/`; job `com.arkb.agentic-tools.core-v2`. The
pipeline waits for the shared lock and a free GPU, registers, executes from the
frozen source, then runs accounting, the independent audit, grading and analysis.
Completion means all 3,640 attempts accounted, all scoring statuses resolved or
explicitly qualified, passing audits and an English report with a Chinese summary.
Independent human review remains an explicit qualification, never a claimed step.

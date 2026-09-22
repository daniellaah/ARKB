# Agentic tool-selection experiment plan

Version: 1.1 — 2026-09-12. Status: **conditional scheduled execution authorized;
waiting for Phase C completion**.

The user has chosen to wait for BrowseComp-Plus preparation and include it in
this experiment. In a subsequent instruction, the user authorized checks starting
on 2026-09-13 at 04:30 America/Los_Angeles, every 30 minutes thereafter, and
automatic commencement of this experiment once Phase C is complete. That
instruction satisfies the earlier requirement for a later execution instruction;
do not ask for that same authorization again.

Codex task `arkb-agentic` returns to this conversation. Its initial 04:30 wakeup
changes the same scheduled task to a 30-minute interval before checking readiness.
It persists orchestration state, prevents duplicate launches and pauses itself
after experiment completion. The new Agent implementation, pilot, core inference
and judge work have not started at the time of this amendment. The existing
Phase C preparation, validation and archival pipeline continues independently.

## 1. Objective and questions

Determine whether ARKB benefits when an Agent can choose exact matching, BM25,
Semantic and Hybrid, reformulate queries, read evidence and decide when to stop.
Compare that behavior with restricted-tool Agents and fixed retrieval followed
by answer generation.

The experiment answers three distinct questions:

1. Does access to all retrieval tools improve answers and evidence acquisition
   compared with the same Agent restricted to one retrieval capability?
2. Does iterative evidence gathering improve on one-pass Semantic or Hybrid RAG,
   and what are its inference and latency costs?
3. Where do failures occur: candidate discovery, evidence delivery, tool choice,
   reading, answer synthesis, citations or termination?

Do not assume that more tools help, or attribute any gain over one-pass RAG
solely to tool choice. That comparison also changes the number of searches and
model steps. Findings apply to the registered model, budgets and datasets; a
single-model study does not establish a universal Agent advantage.

## 2. Scope and existing capabilities

The current runtime already exposes `match`, `search`, `read` and `finish`.
`search(mode=...)` accepts BM25, Semantic or Hybrid. The default is Semantic;
mode selection belongs to the Agent on every call.

`match` is ARKB's bounded exact-match interface. Case-sensitive literal matching
currently uses Python string matching; supported regex/case-insensitive paths
use `rg`. This study evaluates that product interface, not an unrestricted shell
or coding Agent. Renaming search modes into separate functions, adding arbitrary
shell commands, changing match semantics, model selection, chunking, reranking,
fusion weights or Agent stopping policy is outside this first experiment.

Keep the accepted Phase A tool contract, Phase B reranker implementation and
Phase C retained C0 fusion. Reranking is **off in every arm**. Changes needed for
this study belong in evaluation adapters and runners; do not modify production
Agent/retrieval source or the running Phase C scripts.

The earlier [Agent versus baselines report](../evaluation/agent-vs-baselines.md)
uses accumulated Agent evidence versus one-request rankings under different
budgets. It does not settle the controlled tool-selection question. The recent
[validation diagnostics](phase-c-validation-diagnostics.md) motivate investigation
but must not be used to cherry-pick queries or tune the new arms.

## 3. Start conditions

All conditions are checked by the scheduled continuation before new execution:

- Phase C BrowseComp-Plus embedding is complete and its full index is READY:
  100,195 documents, 1,883,193 chunks, 1,847,403 unique embedding inputs under the
  current frozen configuration. Verify actual manifest identities, not just a
  status string or these counts.
- Phase C full validation, replay and archival have completed. There are no
  competing preparation, indexing, evaluation or diagnostic inference jobs on
  the local GPU. Preserve its frozen report and experimental decision.
- FiQA and NFCorpus snapshots and manifests verify; any MuSiQue per-context
  snapshots are verified or prepared once before timing the Agent runs.
- All arms can access the identical corpus revision. Search uses the fixed index;
  match/read use the corresponding unchanged document files. Pin paths and hashes
  before/after runs. Keep the external volume mounted and check free space.
- The evaluator can enforce tool restrictions, evidence limits, model options
  and final-output validation, and its synthetic boundary tests pass.
- Answer labels, judge identity, query selection, prompts, budgets and scoring
  code have an executable, hashed protocol. A missing answer grader must not be
  replaced silently with retrieval recall.

Use verified read-only snapshots or isolated restorations. A restoration may
change a service endpoint, but must preserve vector/payload/document identity.
Do not rebuild full document embeddings merely to run these new Agent queries.

## 4. Experimental arms

| ID | Retrieval access | Interaction |
| --- | --- | --- |
| F-S | Semantic, original question, one retrieval | One answer-generation request |
| F-H | Hybrid, original question, one retrieval | One answer-generation request |
| A-M | `match`, `read`, `finish` | Agent chooses patterns, reads and subsequent actions |
| A-B | `search(mode=bm25)`, `read`, `finish` | Same Agent loop, BM25 only |
| A-S | `search(mode=semantic)`, `read`, `finish` | Same Agent loop, Semantic only |
| A-H | `search(mode=hybrid)`, `read`, `finish` | Same Agent loop, Hybrid only |
| A-All | `match`, all search modes, `read`, `finish` | Same Agent loop, Semantic default |

Primary tool contrasts are A-All minus each of A-M/A-B/A-S/A-H. A-All versus A-S
also keeps the default retrieval mode unchanged. Report all four contrasts; do
not select the strongest-looking comparison after scoring. A-All versus F-S/F-H
are secondary architecture/cost contrasts.

A-M must be able to select its own patterns. Feeding the original natural-language
question verbatim into literal matching would be a different, artificially weak
baseline. All Agent arms share the same read interface, source visibility,
reference validation, error recovery and finalization behavior.

Use one shared instruction template. Render only its capability-availability
paragraph from each arm's schemas; remove hints naming unavailable tools. Freeze
the base template and every rendered prompt. The existing product system prompt
must not accidentally tell a restricted arm to call a hidden tool. Any prompt
adaptation stays in the evaluation request adapter and is identical in all
other respects, including evidence-only answering and stopping instructions.

**Capability enforcement:** setting `AgentTools(mode="hybrid")` only changes a
default. It does not prohibit other modes. Filter tool schemas and mode enums in
an evaluation-only adapter, set an allowed default, and validate at ToolSession's
execution boundary. Omitted/null mode, invalid names, multi-call turns and direct
read selectors must not bypass the arm restriction. Retain rejected attempts
and charge requests/executions according to the documented observer semantics.

F-S/F-H call the existing retrieval engine with top 20 chunks, original question,
unchanged production candidate depth and rerank off. They receive a deterministic
rank-prefix packing of whole evidence chunks within the shared evidence ceiling;
stop at the first chunk that does not fit. Retain every packing decision. There
is no query rewrite, extra read, label-based selection or deduplication policy
introduced solely for these arms. Use the common canonical answer schema and
reference registry, not the legacy generation path with a different thinking
setting or output contract.

## 5. Datasets and fixed query selection

The first measured study includes all four tracks below. Counts are a planned
query sample, not a reduced document corpus. They do not amend Phase C's separate
full-query validation.

| Track | Corpus/context | Core selection | Pilot, excluded from core | Main use |
| --- | --- | ---: | ---: | --- |
| BrowseComp-Plus | Full 100,195-document snapshot; 830 official queries available | 100 queries | 4 queries | Primary end-to-end answer and evidence evaluation |
| FiQA | Full intended corpus, retaining the recorded 38-blank-document indexing exception and original qrels | 50 queries | 2 queries | Evidence coverage and tool behavior |
| NFCorpus | Full 3,633-document snapshot | 50 queries | 2 queries | Evidence coverage and tool behavior |
| MuSiQue | Each variant's original supplied context | 30 original pairs / 60 variants | 3 pairs / 6 variants | Multi-hop answers, support and answerability |

The existing MuSiQue selection contains 100 complete pairs: 34 two-hop, 33
three-hop and 33 four-hop pairs. Sample 10 core pairs and one pilot pair within
each hop stratum. Keep the answerable/unanswerable variants together throughout
selection, scheduling, analysis and bootstrap. This remains a supplied-context
benchmark, not large-corpus open retrieval.

Selection is deterministic and outcome-independent. Within each dataset/stratum,
sort query IDs (MuSiQue original pair IDs) by the SHA-256 hex digest of UTF-8
`agentic-tools-v1|20260912|DATASET|STRATUM|ID`, breaking hash ties by ID. STRATUM is
`all` for the first three datasets and `2`, `3` or `4` for MuSiQue. Take the first
pilot count, then the next core count. Preserve all remaining IDs as reserves.
Generate and hash an explicit ID manifest before inference. Do not inspect
questions, answer strings or ARKB outcomes to choose the sample.

Existing BRIGHT and native fixtures remain useful for harness regression and
historical context; they are not extra main arms or pooled scores in this first
study. They can be included in a separately registered extension. Public labels
and past retrieval results are already exposed, so none of these results should
be described as evaluation on a private, unseen production test set.

## 6. Shared model, budgets and evidence handling

These are the planned v1 settings. An execution-only pilot may reveal a protocol
failure; any change requires a versioned amendment and rerunning the complete
pilot under one setting before core inference. Do not change settings based on
pilot answer scores or adapt budgets by dataset/arm.

| Setting | Planned value / rule |
| --- | --- |
| Answer/Agent model | Existing `qwen3.5:4b`, exact digest and template pinned |
| Sampling | Temperature 0, `think=True` across all seven arms |
| Context / output limit | `num_ctx=32768`, `num_predict=4096` per request, applied through a common evaluation client adapter |
| Agent turns | 8 total, including the current reserved finalization behavior |
| Agent tool/query/read limits | 12 / 10 / 6 using existing observer semantics |
| Delivered evidence ceiling | 8,000 reference-tokenizer body tokens for every arm |
| Elapsed limits | 300 seconds cooperative budget; 360 seconds harness deadline per trial |
| Tool defaults | Current defaults, including limit 5; unchanged Hybrid candidate depth 20 |
| Fixed RAG | One original-question retrieval, top 20 chunks, common answer format and model options |
| Repetitions | Three per query/variant in every arm, including fixed-RAG generation |
| Execution | One active inference trial at a time; rotate arm order |

The 8,000-token evidence setting is an explicit experiment budget, not a change
to the historical 4,000-token runs. Repeated snippets consume allowance as they
do today. Agent tool responses are delivered or withheld as whole observations
by the existing observer; no silent truncation, summary or relevance-based
packing is added. An oversized read remains an observed budget failure.

Pin the reference tokenizer separately from the chat model tokenizer. Reference
JSON/body token counts do not equal provider-rendered prompt tokens. Verify
accepted effective model options and context behavior before core trials; retain
actual prompt/output usage and detect overflow/truncation. A provider rejecting
options or silently truncating inputs is a protocol failure to resolve before
scoring, not a reason to run that arm with different settings.

Agent arms have equal ceilings, not necessarily equal consumption. A match limit
counts occurrences while search returns chunks; a Hybrid call invokes two
retrieval legs. Report both tool calls and underlying retrieval work. Fixed RAG
intentionally has fewer model steps. Do not claim compute-matched gains from
these ceilings alone; report quality and actual cost together.

## 7. Answer labels, scoring and review

BrowseComp-Plus is the primary answer-quality track. Its current Phase C adapter
intentionally projects out answer fields. Implement a separate scoring-side
loader for official answers using the same pinned source revision. Keep answers,
aliases, support labels and judge prompts outside indexed documents, Agent
messages, filenames and tool-visible metadata. Hash scoring inputs separately.

Use the benchmark's published answer-correctness rubric. The
[official judge guide](https://github.com/texttron/BrowseComp-Plus/blob/main/docs/llm_as_judge.md)
uses Qwen3-32B. Prefer that grader and pin model weights, precision, runtime,
prompt and parser. Availability and resource needs are checked later; no grader
is installed by this plan. If its backend or precision differs from the official
setup, label the result an ARKB adaptation and disclose the difference. Do not
silently substitute the 4B answering model as its own judge or use a paid API.

The primary BrowseComp score is answer success over **all attempted trials**:
a valid canonical final result with a correct answer under the frozen rubric.
Execution errors, absent/malformed finals and nonanswers score zero. A valid
answer from reserved finalization can still succeed; record its budget/stop
reason separately. Judge availability/parse failures are scoring failures,
recorded as pending rather than mislabeled Agent errors. Retry the unchanged
judge request up to twice; unresolved labels block a definitive aggregate.

Use identical blinded grading for every arm. Calibrate on all four pilot
BrowseComp queries and both variants of the two-hop pilot MuSiQue pair, across
all seven arms (six scenarios / 42 responses), with
independent human labels and adjudication of disagreements. Record raw agreement,
confusion counts and uncertainty; 42 cases are calibration, not proof of a
perfect judge. Grader changes use pilot data only, require new version/hashes
and finish before core grading. If independent review is unavailable, retain
judge scores as provisional and report evidence/cost results separately.

For citation support, select the first 20 core BrowseComp query IDs by the same
hash order and the first trial from every arm: 140 responses. Blind arm identity
and check whether each substantive answer claim follows from its cited excerpts.
Report supported-claim fraction and response-level unsupported-claim rate, with
empty/abstained responses separate. Log quote/span provenance. This sample audits
faithfulness; it does not confer human verification on all core responses.

MuSiQue uses the current canonical final-output adapter and official-style answer
EM/F1, support F1, answerability and pair-level sufficiency metrics. Gold aliases
are scorer-only. Missing/invalid outputs do not earn abstention credit. Verify
mapping with synthetic fixtures, including partial and insufficient-evidence
outputs, without repairing model prose using the gold answer.

FiQA/NFCorpus support **evidence diagnostics**, not standalone answer correctness:
qrels identify relevant sources, not a complete answer rubric. Any answer review
on these tracks must be labeled exploratory and separately annotated.

## 8. Measurements and statistical analysis

Preserve complete traces, then compute:

| Outcome | Measures |
| --- | --- |
| Answer quality | BrowseComp answer success; MuSiQue answer/support/answerability metrics; citation audit |
| Evidence | Positive document recall, evidence/gold label sets separately for BrowseComp, aspect/support coverage where available |
| Tool behavior | Calls and errors by tool/mode, mode transitions, first positive discovery, new versus repeated sources, read expansion, termination reasons |
| Resource use | Input/output tokens, model requests, query/read calls, internal retrieval legs, delivered evidence tokens, elapsed p50/p95, model-load/setup costs separately |
| Reliability | All attempted trials, malformed outputs, invalid references, budget stops, timeouts and between-trial variation |

Keep evidence that was returned, delivered to conversation and submitted to a
subsequent model request distinct. Submitted evidence does not prove model
comprehension or provider processing after a failed request. Report coverage on
all attempts and, separately, successful-final subsets; never drop failures from
the main denominator. Unjudged documents are unknown rather than verified
negatives. Cumulative Agent evidence is not one ranked list: do not manufacture
Agent nDCG by concatenating successive searches.

Average repetitions within a query before dataset aggregation. For MuSiQue,
keep whole original pairs, preserving the three hop strata. Use paired query
bootstrap with 20,000 resamples and seed 20260912, same sampled units across arms.
Report differences, intervals, wins/ties/losses and raw sample sizes; repeated
trials do not triple the number of independent questions.

The four primary BrowseComp tool contrasts form one comparison family. Report
nominal 95% intervals for readability and Bonferroni-adjusted 98.75% bootstrap
intervals for the four-comparison decision (percentiles 0.625 and 99.375).
These bootstrap coverage guarantees are approximate with finite samples.
Secondary architecture comparisons, other datasets and behavioral slices are
exploratory, with no pooled cross-dataset score or retrospective best-arm choice.

A practically meaningful improvement over a particular restricted Agent requires
an observed answer-success gain of at least 5 percentage points and its adjusted
interval entirely above zero. Claim advantage over every restricted arm only if
all four contrasts meet that rule. Otherwise report mixed, no demonstrated gain
or inconclusive evidence as appropriate. Also report token and latency ratios;
higher quality is not automatically better cost-effectiveness.

A 100-query BrowseComp sample may not resolve small effects. A wide interval is
an inconclusive result, not evidence of equality. Any extension of query counts
must be decided using blinded precision/cost planning and registered before new
outcomes are viewed; do not keep adding cases until a favorable interval appears.
No automatic product configuration switch follows this study.

## 9. Pilot, run order, cost and interruption rules

After data readiness (conditional execution is already authorized above):

1. Implement the evaluation adapters and synthetic contract checks. Verify tool
   restrictions, label isolation, shared answer formatting, evidence accounting
   and failure denominators. Prepare/reuse identical snapshots before timing.
2. Freeze selection IDs, all seven arms, model/runtime options, proposed budgets,
   prompts and pilot manifests. Run all 14 pilot scenarios once per arm: **98
   trajectories**. Keep pilot outputs separate from core evidence.
3. Use the pilot to verify execution, context handling and resource estimates.
   Calibrate the scorer separately. Any technical amendment applies to every
   arm and is recorded before rerunning the pilot; it cannot use core outcomes.
4. Freeze the final executable protocol and run all core cases in balanced order:
   **260 scenarios × 7 arms × 3 repetitions = 5,460 trajectories**. RAG evidence
   can be cached after its first retrieval per query; repeat generation and
   report retrieval and generation costs separately.
5. Run frozen scoring, blinded citation review, replay/statistical checks and
   produce a report including all arms and attempts. Preserve indexes and raw
   artifacts for subsequent independent replay.

Use a deterministic cyclic arm rotation based on query index and repetition
modulo seven. MuSiQue paired variants share each rotation. Separate warmups from
measured trials; record cold-load events and any service restarts. Runtime traces
under a shared GPU are not clean latency comparisons. Do not run competing
embedding/backend probes during this measured experiment.

Estimate total runtime after the pilot as the sum, over dataset and arm, of
core cases × repetitions × observed mean seconds, plus measured preparation,
scoring and archival. For scale only, an overall average of 15/30/60 seconds per
trajectory would imply approximately 22.8/45.5/91 hours for core inference alone.
These are capacity examples, not measured ETAs. Judge work (2,100 core BrowseComp
answers before retries/review) and MuSiQue preparation are additional. Publish
this estimate before core dispatch; do not hide sample reduction or model changes
as performance optimizations.

Persist one record per attempt keyed by protocol hash, dataset, query/variant,
arm and repetition. Completed attempts are immutable. On interruption, verify
all identities and skip completed keys. Model/tool timeouts count as outcomes;
only clearly identified infrastructure failures may have a recorded replacement,
with the original retained and both first-attempt and replacement sensitivity
results shown. Do not repeatedly rerun poor answers or discard unsuccessful arms.

## 10. Required implementation and deliverables

The seven-arm runner does not exist yet. Future implementation should produce
these evaluation-only components (names are proposed, not current commands):

- Capability-restricted Agent adapter and one-pass RAG adapter sharing model,
  evidence-reference and final-output contracts.
- Manifest/selection builder, read-only snapshot loader, sequential resumable
  runner and model/service fingerprint checks.
- Scoring-only BrowseComp answer importer, pinned judge adapter, blinded review
  export and strict MuSiQue mapping.
- Offline trace replay, evidence/cost aggregation, paired statistics and report
  generation. Boundary fixtures must reject arm escape and gold leakage.

Use a new experiment identity such as `agentic-tools-v1`. Small protocols,
aggregate summaries and replay checks belong under
`evaluation/agentic-tools/v1/`; large raw traces, decrypted BrowseComp inputs,
judge outputs and snapshots stay in a dedicated directory on the approved
external volume. Public-facing reports contain aggregate results and opaque
IDs, not decrypted BrowseComp questions/answers. Preserve every checksum and
source/configuration copy; do not overwrite P4 or Phase A/B/C artifacts.

Expected outputs: `protocol.json`, `selection.json`, `inputs.json`, measured
source/config hashes, append-only attempt records, answer/evidence scores,
review records, `summary.json`, paired intervals, failure taxonomy, cost report,
replay verification and an English Markdown report with a Chinese user summary.

Completion requires all registered attempts accounted for, all scoring statuses
resolved or explicitly qualified, matched data/model configurations, passing
trace/metric audits, and every baseline shown. The report must state which
contrasts improve, which regress, which are inconclusive and whether the extra
cost is justified for the measured tasks. No deferred stage is described as done.

## 11. Items resolved at execution time

This plan fixes the intended design without pretending that execution artifacts
already exist. Before core inference, record the actual snapshot IDs and hashes,
query/pair IDs, chat and judge model revisions/precision, effective context
behavior, scorer prompt/parser hashes, reviewer assignment, pilot measurements,
estimated runtime/storage and any versioned technical amendment. If compute or
review capacity is unavailable, retain the plan and report the specific unmet
condition; do not quietly run a different experiment.

# Agentic versus fixed-workflow retrieval on one simple architecture (v1)

Question: on the same architecture, model and evidence budget, what does letting
the model decide its retrieval actions (fully agentic) change in outcomes and
cost against a fixed "one retrieval, one generation" workflow?

Setup (2026-09-20 UTC, commit 942cdc4): four arms from
`evaluation/agentic_tools/contract.py`, Qwen3.5-9B with thinking, temperature
0, 32,768 context, 4,096 output tokens per request, at most 8 turns, budgets
12/10/6 tool, query and read calls, 8,000 evidence tokens and 300 s. F-S and
F-H run one `search(limit=20)` (semantic or hybrid) on the original question,
pack whole chunks in rank order up to 8,000 tokens and generate the final
object once, with thinking. A-S has `search(semantic)`, `read` and `finish`;
A-All adds `match` and the choice of bm25, semantic or hybrid. Agent arms
think during tool turns and finalize without thinking, as the product does.
Every arm receives the same answer-format sentence (short answer on the first
line). Data: the 144-scenario development set (v2 60, exact-nfcorpus 24,
recall-nfcorpus 20, recall-fiqa 20, musique 20) plus the optional ten-query
long-document slice. Runs `aw-9b-think-<ARM>`, `aw-long-9b-think-<ARM>` and
`aw-4b-nothink-<ARM>` under `evaluation/results/devloop/`;
comparisons under `evaluation/devloop/comparisons/aw-*.md`.

## Results by slice and arm

| Slice / metric | n | F-S | F-H | A-S | A-All |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 span coverage (delivered) | 60 | 0.968 | 0.968 | 0.957 | 0.968 |
| v2 cited source recall | 60 | 0.712 | 0.567 | 0.856 | 0.894 |
| v2 answered / no_retrieval / abstained | 60 | 28/60 · 0/4 · 2/4 | 23/60 · 0/4 · 2/4 | 40/60 · 3/4 · 4/4 | 42/60 · 3/4 · 4/4 |
| exact completeness (cited) | 24 | 0.016 | 0.057 | 0.167 | 0.875 |
| exact complete-and-exact | 24 | 0.000 | 0.000 | 0.000 | 0.875 |
| exact spurious cited (mean) | 24 | 0.000 | 0.000 | 0.458 | 0.000 |
| NFCorpus recall (delivered) | 20 | 0.201 | 0.190 | 0.166 | 0.166 |
| FiQA recall (delivered) | 20 | 0.505 | 0.462 | 0.400 | 0.400 |
| MuSiQue support F1 | 20 | 0.460 | 0.395 | 0.560 | 0.654 |
| MuSiQue answerability | 20 | 0.500 | 0.350 | 0.700 | 0.650 |
| MuSiQue answer F1 (first line) | 20 | 0.325 | 0.250 | 0.675 | 0.600 |
| MuSiQue answer EM (first line) | 20 | 0.300 | 0.200 | 0.500 | 0.600 |
| MuSiQue answer F1 (full text) | 20 | 0.325 | 0.250 | 0.675 | 0.600 |
| v2 elapsed s / error finals | 60 | 32.3 / 14 | 39.6 / 19 | 15.9 / 0 | 14.6 / 0 |
| exact-nfcorpus elapsed s / error finals | 24 | 63.5 / 15 | 67.7 / 15 | 65.1 / 4 | 32.9 / 3 |
| recall-nfcorpus elapsed s / error finals | 20 | 42.7 / 4 | 44.9 / 5 | 32.4 / 0 | 31.1 / 0 |
| recall-fiqa elapsed s / error finals | 20 | 38.2 / 4 | 32.0 / 2 | 30.0 / 0 | 29.3 / 0 |
| musique elapsed s / error finals | 20 | 38.9 / 5 | 47.3 / 9 | 48.4 / 2 | 50.0 / 1 |

Delivered coverage and recall count evidence that entered the conversation,
so a failed generation does not lower them; cited, answered and answer
metrics score the final object, and an error final scores 0 (or "not
answered"). Error finals: F-S 42, F-H 50, A-S 6, A-All 4 of 144. In the fixed
arms 39 and 49 of these are one mechanism: the single generation, asked to
think and then emit the JSON object, spent the whole 4,096-token output
allowance thinking about 20 packed chunks and returned nothing (F-H: 18 of 60
v2 questions across every task type, 15 of 24 enumerations, 7 of 40 recall
queries, 9 of 20 MuSiQue). The agent arms' errors are the same overflow inside
a tool turn (A-All 4, A-S 3) plus three invalid finals in A-S.

## Paired differences (per question, slice-stratified bootstrap, nominal 95%)

Primary contrast A-All minus F-H, then the secondary contrasts; `*` marks an
interval that excludes zero; counts are wins/ties/losses for the left arm.
Seed 20260912, 20,000 resamples, no multiplicity correction.

| Slice | metric | n | A-All minus F-H | A-All minus F-S | A-S minus F-S | A-All minus A-S |
| --- | --- | ---: | --- | --- | --- | --- |
| v2 | evidence_coverage_delivered | 47 | +0.000 [+0.000, +0.000] 0/47/0 | +0.000 [+0.000, +0.000] 0/47/0 | -0.011 [-0.032, +0.000] 0/46/1 | +0.011 [+0.000, +0.032] 1/46/0 |
| v2 | source_recall_cited | 52 | +0.327 [+0.173, +0.481]* 22/25/5 | +0.183 [+0.029, +0.337]* 15/30/7 | +0.144 [-0.010, +0.298] 14/31/7 | +0.038 [-0.048, +0.135] 5/43/4 |
| v2 | answered | 60 | +0.317 [+0.200, +0.450]* 20/39/1 | +0.233 [+0.100, +0.367]* 17/40/3 | +0.200 [+0.033, +0.367]* 20/32/8 | +0.033 [-0.067, +0.133] 6/50/4 |
| v2 | no_retrieval_respected | 4 | +0.750 [+0.250, +1.000]* 3/1/0 | +0.750 [+0.250, +1.000]* 3/1/0 | +0.750 [+0.250, +1.000]* 3/1/0 | +0.000 [+0.000, +0.000] 0/4/0 |
| v2 | abstained | 4 | +0.500 [+0.000, +1.000] 2/2/0 | +0.500 [+0.000, +1.000] 2/2/0 | +0.500 [+0.000, +1.000] 2/2/0 | +0.000 [+0.000, +0.000] 0/4/0 |
| exact-nfcorpus | completeness_cited | 24 | +0.818 [+0.657, +0.949]* 21/1/2 | +0.859 [+0.713, +0.988]* 21/2/1 | +0.151 [+0.061, +0.256]* 9/15/0 | +0.708 [+0.523, +0.870]* 21/1/2 |
| exact-nfcorpus | spurious_cited | 24 | +0.000 [+0.000, +0.000] 0/24/0 | +0.000 [+0.000, +0.000] 0/24/0 | +0.458 [+0.042, +0.958]* 4/20/0 | -0.458 [-0.958, -0.042]* 0/20/4 |
| recall-nfcorpus | positive_recall_delivered | 20 | -0.024 [-0.055, -0.002]* 1/13/6 | -0.035 [-0.068, -0.010]* 1/10/9 | -0.036 [-0.069, -0.010]* 1/10/9 | +0.001 [+0.000, +0.002] 1/19/0 |
| recall-fiqa | positive_recall_delivered | 20 | -0.062 [-0.148, +0.017] 1/15/4 | -0.105 [-0.216, +0.000] 2/11/7 | -0.105 [-0.235, +0.000] 1/14/5 | -0.000 [-0.117, +0.142] 2/15/3 |
| musique | support_f1 | 10 | +0.259 [+0.000, +0.537]* 6/2/2 | +0.194 [-0.060, +0.493] 4/3/3 | +0.100 [-0.227, +0.440] 3/4/3 | +0.094 [-0.047, +0.251] 4/4/2 |
| musique | answerability_correct | 20 | +0.300 [+0.100, +0.500]* 6/14/0 | +0.150 [-0.100, +0.400] 5/13/2 | +0.200 [+0.000, +0.400] 5/14/1 | -0.050 [-0.300, +0.200] 3/13/4 |
| musique | answer_f1_first_line | 10 | +0.350 [+0.100, +0.650]* 4/6/0 | +0.275 [+0.000, +0.575] 3/7/0 | +0.350 [+0.075, +0.625]* 4/6/0 | -0.075 [-0.225, +0.000] 0/9/1 |
| musique | answer_em_first_line | 10 | +0.400 [+0.100, +0.700]* 4/6/0 | +0.300 [+0.000, +0.600] 3/7/0 | +0.200 [+0.000, +0.500] 2/8/0 | +0.100 [+0.000, +0.300] 1/9/0 |
| all (stratified) | elapsed_s | 144 | -18.1 [-22.0, -14.2]* 40/0/104 | -13.8 [-17.6, -10.1]* 52/0/92 | -7.8 [-11.8, -4.0]* 72/0/72 | -6.0 [-8.3, -3.5]* 51/0/93 |
| all (stratified) | model_requests | 144 | +3.1 [+2.9, +3.4]* 143/1/0 | +3.1 [+2.9, +3.4]* 143/1/0 | +3.2 [+3.0, +3.4]* 143/1/0 | -0.1 [-0.3, +0.1] 30/75/39 |
| all (stratified) | tool_calls | 144 | +2.2 [+1.9, +2.5]* 109/32/3 | +2.2 [+1.9, +2.5]* 109/32/3 | +2.8 [+2.5, +3.0]* 131/10/3 | -0.6 [-0.8, -0.3]* 29/60/55 |
| all (stratified) | prompt_tokens | 144 | +11,214 [+9,177, +13,341]* 108/0/36 | +11,437 [+9,409, +13,556]* 110/0/34 | +14,275 [+12,150, +16,513]* 131/0/13 | -2,837 [-5,132, -471]* 87/0/57 |
| all (stratified) | eval_tokens | 144 | -464 [-767, -169]* 97/0/47 | -194 [-485, +84] 106/0/38 | +16 [-278, +293] 108/0/36 | -210 [-325, -93]* 50/0/94 |

On the questions where the fixed generation did complete (descriptive only,
this subset is selected by the fixed arm's own outcome):

| Slice | metric | questions where F-H completed | F-H | A-All | diff |
| --- | --- | ---: | ---: | ---: | ---: |
| v2 | source_recall_cited | 35 of 60 | 0.843 | 0.900 | +0.057 |
| v2 | answered | 41 of 60 | 0.561 | 0.756 | +0.195 |
| exact-nfcorpus | completeness_cited | 9 of 24 | 0.151 | 0.778 | +0.627 |
| musique | answer_f1_first_line | 6 of 20 | 0.417 | 0.500 | +0.083 |
| musique | answerability_correct | 11 of 20 | 0.636 | 0.727 | +0.091 |
| musique | support_f1 | 6 of 20 | 0.658 | 0.623 | -0.035 |

## Cost per question

| Cost (per question, all slices) | F-S | F-H | A-S | A-All |
| --- | ---: | ---: | ---: | ---: |
| elapsed s | 40.7 | 45.0 | 32.8 | 26.9 |
| model requests | 1.00 | 1.00 | 4.23 | 4.15 |
| tool calls | 1.00 | 1.00 | 3.75 | 3.19 |
| prompt tokens | 6,107 | 6,330 | 20,382 | 17,545 |
| output tokens | 1,224 | 1,494 | 1,240 | 1,030 |
| delivered evidence tokens | 2,919 | 3,131 | 3,162 | 2,082 |
| final status error | 42 | 50 | 6 | 4 |
| responses cut at num_predict | 39 | 49 | 3 | 4 |
| wall minutes | 98 | 108 | 79 | 65 |
| statuses | answered 46, error 42, insufficient_evidence 49, partial 7 | answered 38, error 50, insufficient_evidence 46, partial 10 | answered 76, error 6, insufficient_evidence 62 | answered 85, error 4, insufficient_evidence 50, partial 5 |

Token counts are Ollama's `prompt_eval_count` and `eval_count` summed over a
question's requests; the fixed arms' output count is dominated by the
overflowed generations (4,096 each).

Long-document slice (ten BrowseComp-Plus queries, same budgets):

| long-browsecomp (n=10) | F-S | F-H | A-S | A-All | A-All minus F-H |
| --- | ---: | ---: | ---: | ---: | --- |
| positive recall (delivered) | 0.064 | 0.130 | 0.070 | 0.013 | -0.118 [-0.227, -0.027] 1/3/6 |
| elapsed s | 57.3 | 61.9 | 41.3 | 39.0 | -22.8 [-38.3, -6.7] |
| finals | error 5, insufficient_evidence 5 | error 5, insufficient_evidence 5 | insufficient_evidence 10 | insufficient_evidence 10 | |
| overflowed generations | 5 | 4 | 0 | 0 | |
| delivered evidence tokens | 7,755 | 7,814 | 7,779 | 7,778 | |
| model requests / prompt tokens | 1.0 / 11,120 | 1.0 / 11,502 | 4.1 / 32,024 | 3.9 / 31,746 | |

Cost reference, Qwen3.5-4B without thinking (F-H and A-All only, 144 questions):

| 4B no thinking | n | F-H | A-All | A-All minus F-H |
| --- | ---: | ---: | ---: | --- |
| exact-nfcorpus completeness_cited | 24 | 0.114 | 1.000 | +0.886 [+0.785, +0.959]* 23/1/0 |
| musique support_f1 | 10 | 0.605 | 0.347 | -0.258 [-0.492, +0.000] 1/3/6 |
| musique answerability_correct | 20 | 0.450 | 0.550 | +0.100 [-0.200, +0.400] 6/10/4 |
| musique answer_f1_first_line | 10 | 0.483 | 0.181 | -0.303 [-0.633, +0.077] 2/2/6 |
| recall-fiqa positive_recall_delivered | 20 | 0.462 | 0.415 | -0.047 [-0.187, +0.075] 2/14/4 |
| recall-nfcorpus positive_recall_delivered | 20 | 0.190 | 0.141 | -0.049 [-0.127, +0.001] 2/11/7 |
| v2 source_recall_cited | 52 | 0.913 | 0.817 | -0.096 [-0.212, +0.019] 4/39/9 |
| v2 answered | 60 | 0.650 | 0.750 | +0.100 [-0.033, +0.233] 12/42/6 |
| all (stratified) elapsed_s | 144 | 5.8 | 10.1 | +4.2 [+3.3, +5.2]* 116/0/28 |
| all (stratified) prompt_tokens | 144 | 5,416 | 14,083 | +8,668 [+7,034, +10,419]* 108/0/36 |
| error finals / overflowed responses / wall min | 144 | 6 / 0 / 15 | 1 / 0 / 24 | |

## Reading

1. **Primary contrast (A-All minus F-H).** Evidence delivery is not where the
   arms differ: v2 span coverage is identical (0.968, 0 wins, 0 losses) and the
   agent delivers *less* on the recall tracks (FiQA -0.062 [-0.148, +0.017],
   NFCorpus -0.024 [-0.055, -0.002]). The final object is where the agent wins:
   cited source recall +0.33 [+0.17, +0.48], answered +0.32 [+0.20, +0.45]
   (42 against 23 of 60), enumeration completeness +0.82 [+0.66, +0.95],
   MuSiQue answerability +0.30 [+0.10, +0.50] and first-line answer F1 +0.35
   [+0.10, +0.65]. Cost: the agent was *faster* here (-18 s per question,
   26.9 against 45.0 s), at 3.1 more model requests, 2.2 more tool calls and
   2.8 times the prompt tokens (17.5k against 6.3k).
2. **Most of that gap is the fixed arm not finishing, not the agent finding
   more.** With thinking on, the fixed generation overflowed the 4,096-token
   output allowance in 49 of 144 questions (F-S: 39), each costing about 70 s
   and an error final. On the questions where F-H did complete, the agent's
   lead shrinks to +0.06 cited recall, +0.20 answered, +0.08 MuSiQue F1 and
   -0.04 support F1; only enumeration stays large (+0.63), because a fixed
   search cannot list every note containing a literal term.
3. **The 4B nonthinking reference confirms it.** Without thinking neither arm
   overflows, and the fixed workflow is then equal or better on every metric
   except enumeration (v2 cited recall 0.913 against 0.817, MuSiQue support F1
   0.605 against 0.347, first-line F1 0.483 against 0.181, recall within
   noise), while the agent costs 1.7 times the time and 2.6 times the prompt
   tokens. Enumeration remains the one clear agent win (1.000 against 0.114).
4. **Iteration substituted for depth.** The 9B agent made 1.3 to 1.8 searches
   per recall question at the default depth of ten and delivered about 12
   distinct documents, against 20 for the fixed arm at `limit=20`; delivered
   recall tracks the engine's recall@10 for the agent (FiQA 0.40) and
   recall@20 for the fixed arm (0.46 hybrid, 0.51 semantic). More turns did
   not buy more coverage of the positives.
5. **Tool and mode choice were barely exercised.** A-All never chose bm25 or
   hybrid (15 explicit `semantic` in 232 searches, the rest default) and used
   `match` almost only on enumeration (1.25 per exact question, 0 to 0.25
   elsewhere). A-All minus A-S is inside noise on every non-enumeration
   metric; on enumeration `match` is worth +0.71 [+0.52, +0.87] completeness,
   and A-S, forced to enumerate by search and read, cites 0.46 spurious notes
   per question against none for A-All. On this set the value of "full
   agentic" over "single-tool agent" is one tool, not the choice among tools.
6. **What iteration reliably bought at 9B:** the behavioural cases. The
   arithmetic and sorting tasks were answered without retrieval (3 of 4
   against 0 of 4), all four unanswerable v2 questions were abstained on
   (4 of 4 against 2 of 4), and MuSiQue answerability rose from 0.35 to 0.65
   with no losses (6 wins, 14 ties). Each is a decision the fixed pipeline
   cannot make because it has no turn in which to make it.
7. **Long documents:** every arm ends in `insufficient_evidence` or error and
   fills the 8,000-token evidence allowance within two or three calls; delivered
   recall is 0.01 to 0.13 and the fixed arms lose four to five of ten
   generations to overflow. The agent is cheaper in time (-23 s) but its
   recall is lowest (A-All 0.013, 1 win, 6 losses against F-H). The budget,
   not the control flow, is binding here, as the capacity axis already found.
8. **The answer-format sentence worked.** First-line and full-text MuSiQue F1
   coincide for every arm (the model puts the short answer first), so the
   0.08 answer F1 of the capacity axis was a format artifact; with the same
   9B model the agent arms now score 0.60 to 0.68.

Answer to the question: on this simple architecture, letting the model
iterate did not measurably improve evidence acquisition (identical v2
coverage, slightly lower document recall) and cost about 2.8 times the prompt
tokens and 3 more model requests per question; it improved the *decisions*
around the answer (abstaining, not retrieving, citing what was found, listing
every match). Most of the headline gap against the fixed workflow came from a
configuration defect of the fixed arm (thinking inside one structured
generation), not from autonomy.

## Limits

Development material only: the scenarios are exposed, one run per arm, the
v2 labels are provisional, no human has reviewed any answer (a blinded sheet
with 20 v2 and 10 MuSiQue questions chosen by ID hash is exported under
`evaluation/results/devloop/review/aw-v1/`; **human review pending**). The
repeatability study measured two thirds of cells changing their final object
between identical runs, so single-question wins and losses are noisy; only
the intervals are meant to be read. Intervals are nominal and uncorrected over
about fifty metric-by-slice cells. The fixed arms' overflow is a property of
this configuration (thinking, structured output and 4,096 output tokens in one
request), not of fixed workflows in general; with it, the primary contrast
measures "agent with thinking turns" against "one thinking generation that
often does not finish". The run records carry `dirty` because the README and
the review-sheet script changed in the working tree during the chain; no
imported module differs from commit 942cdc4. Nothing here is a quality claim.

## Next step

Give the fixed workflow the same two-request finalization the agent has (one
thinking request without `format`, one formatting request without thinking)
and rerun F-H only. That removes the overflow mechanism without changing
"one retrieval, one answer", and turns the primary contrast into a measurement
of iteration alone; everything else in this report can be reused as is.

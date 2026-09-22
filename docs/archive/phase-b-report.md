# Phase B: Reranker reliability and long-query robustness

**Completed: select B3 and keep reranking optional.**

## 1. Baseline reproduction

The accepted Phase A production baseline is `c47a015`; the starting clean tree
was `00e9bb7`, which adds the Phase A evaluation evidence. The unchanged
reranker reproduced all **10,320 scores exactly** and all **516 source rankings**
from P4: 300 SciFact queries, 115 Stack Overflow queries and 101 Robotics queries.
The technical domains retain P4's **Bright-Pro** corpus, qrels and supplied
aspect annotations, pinned at `dbdc22babbef310210e267b99249e7cec86d5edf`;
they are not a newly reconstructed classic-BRIGHT evaluation. SciFact retains
the BEIR test split, used here as exposed development data.
The preregistered absolute score tolerance was `1e-4`; the observed maximum
error was **zero** on every dataset. No replacement baseline was selected.

The frozen pool retains full query text, original Hybrid top 100, selected
20-source candidates with complete SearchResults, original scores, both saved
retrieval legs and source/chunk/revision/coordinate provenance. Labels and
aspects are stored separately in `scoring/`; model input construction uses only
query, title and body. No upstream retrieval or indexing was repeated per
variant. All 10,320 source spans and identities were validated at freeze time.
A separate arithmetic audit verified all **51,600 Hybrid source positions**,
including the chosen chunk and exact RRF score for every rerank candidate.

Original P4 artifacts remain intact. SciFact's older P4 run did not record
weight-file hashes or the complete software environment. Its source pins the
same checkpoint, and its full score reproduction is exact; this does not invent
missing historical provenance. The original Stack Overflow and Robotics weight
hashes agree with the current files.

## 2. Reranker architecture

P4 first retrieves 500 chunk candidates from each fixed leg and applies the
existing RRF with `k=60`. It collapses to the first occurrence of each source,
then reranks the first 20 unique sources using each source's best Hybrid-ranked
chunk. This is not necessarily chronological chunk zero: 225/6000 SciFact,
963/2300 Stack Overflow and 460/2020 Robotics selected chunks have a nonzero
chunk index. The source tail at positions 21–100 is appended unchanged.

The general production `RerankedRetriever` instead consumes its retriever's
chunk candidates and applies the requested final `top_k`. It does not implement
P4's source collapse or 100-source tail. Distinct chunks of one source remain
valid candidates; duplicate SearchResult identities are rejected. Phase B
preserves both boundaries.

The model remains `Qwen/Qwen3-Reranker-0.6B`, revision
`e61197ed45024b0ed8a2d74b80b4d909f1255473`, CPU float32, SDPA, batch size 16.
Scores remain final-position `logit(yes) - logit(no)`, saved as lossless JSON
numbers without probability conversion or rounding. The instruction and chat
wrapper are unchanged. The [pinned official model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/e61197ed45024b0ed8a2d74b80b4d909f1255473/README.md)
documents that wrapper and advertises a 32K context; the downloaded configuration
contains `max_position_embeddings=40960`. This experiment uses **512** throughout.

The wrapper reserves 39 prefix and 9 suffix tokens. Including the instruction
and field labels, the actual full-frame template cost in these inputs is at
most 73 tokens. The old path right-truncates one concatenated sequence containing
instruction, query, document label, title and body, then left-pads for scoring.
Thus a long query can remove even the document label and all document evidence.
The old Reranker breaks exact score ties using document/chunk identity.

Stable sorting now preserves incoming order for equal scores. Invalid score
containers, missing/non-finite scores and RuntimeError/OSError at the scoring
boundary preserve incoming candidates, order, methods and scores, with explicit
fallback diagnostics. Configuration/programming errors are not indiscriminately
caught. Upstream source/snapshot errors remain outside this fallback boundary.

Input construction is independently inspectable through `QwenInputBuilder` and
`QwenRerankerScorer.prepare_inputs`. The record includes actual unpadded token
IDs and hash, query/title/body token counts before and after allocation, template
and special-token counts, body retention, truncation and original input rank.
Special-token counts overlap the complete token count; they are not an extra
additive budget. Query-gap tokens count toward the query cap and are reported
separately. Original query/title counts describe the original frame; full body
counts describe the allocated frame before body truncation.

Allocated fields are selected as complete Unicode character spans, inserted
back into the official text template, re-tokenized, and checked against their
caps. Reassembling delimiters matters: simply splicing token groups can remove
newlines absorbed into a field's last BPE token. Returned evidence content and
its source coordinates remain unchanged.

## 3. Root-cause findings

| Old input diagnostic | SciFact | Stack Overflow | Robotics |
| --- | ---: | ---: | ---: |
| Candidate inputs | 6000 | 2300 | 2020 |
| Truncated inputs | 1237 (20.62%) | 1582 (68.78%) | 1140 (56.44%) |
| Zero-body inputs | 0 | 600 (26.09%) | 520 (25.74%) |
| Queries with all bodies absent | 0 | 30 | 26 |
| Queries with at least half the bodies absent | 0 | 30 | 26 |
| Mean retained body tokens | 286.47 | 120.84 | 113.18 |
| Median retained body tokens | 289 | 106 | 85 |
| Queries containing score ties | 0 | 67 | 67 |
| Queries with all scores exactly equal | 0 | 30 | 26 |
| Rankings changed by stable ties alone | 0 | 46 | 53 |

All 5,961 equal-score candidate pairs in Stack Overflow and all 5,041 in Robotics
have identical effective inputs. For the 30 and 26 all-body-empty queries,
all candidate inputs and scores collapse to the same value. This connects the
measured input loss to the loss of useful model preference.

With token-identical diagnostics attached, B0 has 21 Stack Overflow and 18
Robotics queries where a known positive exits the top 10 while retaining fewer
than 32 body tokens. Stable B1/B2 reduces these observed counts to 3 and 0.
The loss association is not summed with other overlapping categories.

B1 is a causal tie-policy replay: only the sort rule changes, with exactly the
same saved scores. Stack Overflow nDCG@10 rises from 0.341227 to 0.398871;
Robotics rises from 0.291856 to 0.348986. These recover about 41% and 58% of the
original losses against Hybrid, while leaving a residual gap. SciFact is
unchanged. Input-loss/demotion co-occurrence is observational; the allocation
experiments measure the combined policy effect, including both retained query
information and document evidence.

There are six additional identical-input pairs across the 16/4 batch boundary
whose float32 scores differ slightly, by at most `3.9101e-5`. Fixed-batch full-run
reproduction is exact. Exact-score tie handling does not introduce an arbitrary
epsilon to merge these near ties.

Query-only inspection found that code/logs often occupy the middle of technical
questions. Among queries longer than 192 tokens, 31/76 Stack Overflow and 25/55
Robotics queries have a question mark in their final quarter. Some inspected
tails instead contain logs. These are text-structure observations, not gold-based
importance labels; they motivate the deterministic head-plus-tail experiment
without establishing that every retained tail is useful.

The diagnostic label “demotion with body below 32 tokens” is an association,
not a causal verdict. A short passage can be complete and sufficient. The saved
before/retained counts distinguish naturally short bodies from truncation loss;
the guaranteed body allocation is `min(full body tokens, 128)`.

Both registered allocations have **zero zero-body inputs**, no length-limit
violations, and no violations of the guarantee to retain at least
`min(full body tokens, 128)`. SciFact changes only two candidate inputs (long
titles). The large changes occur on BRIGHT queries. One Robotics input gives up
one dangling byte token at its cutoff to keep Unicode valid; the old decoded
input ended with a replacement character. No original evidence is modified.

## 4. Experiment matrix

B0 is full unchanged-model reproduction. B1 is saved-score stable-tie replay.
B2 adds token instrumentation while preserving every input token and B1 ranking.
B3 uses query cap 128, title cap 64; B4 uses query cap 192, title cap 64.
Both use the existing 512-token sequence limit, head-plus-tail query selection
with a newline/ellipsis/newline gap, and the remaining space for body. The same
fixed 20-source pool is used throughout. Every B3/B4 query is executed once;
their order alternates per query. No new model, upstream retrieval, larger
context window, prompt tuning or gold-based extraction is introduced.

The preregistration fixes 10,000 paired query bootstrap resamples with seed
20260911, reports separate datasets, and preserves wins/ties/losses. Intervals
are exploratory and are not corrected for multiple comparisons. Latency means
actual per-query reranker wall time, including input construction and excluding
model loading and upstream retrieval. Per-candidate latency is amortized
query latency divided by 20, not independent single-candidate serving time.
B1/B2 reuse saved inference and retain the original P4 timings; they are not
new latency measurements. Fresh B0 mean scorer times are 4.501 s, 4.638 s and
4.611 s for SciFact, Stack Overflow and Robotics respectively.
These are development-machine wall times. Occasional offline audits and small
deterministic tests ran concurrently; this was not an isolated serving benchmark
or a latency SLA measurement.

The first B2 instrumentation pass assigned a boundary-crossing separator token
to an empty title. A focused regression caught this; B2-corrected requires a
nonempty title span. Both attempts are retained. Model tokens, scores and
rankings never changed, and the correction preceded allocation quality trials.

**scifact quality**

In the tables, `aspect_recall@10` is weighted aspect recall using the supplied
aspect weights. `alpha_ndcg@10` retains P4's weighted aspect discount with
alpha = 0.5. Metric definitions are unchanged.

| Variant | ndcg@10 | recall@10 | recall@100 | Mean reranker ms/query |
| --- | --- | --- | --- | --- |
| Hybrid | 0.721640 | 0.854556 | 0.963333 | — |
| B0 | 0.766586 | 0.885889 | 0.963333 | 4500.9 |
| B1 | 0.766586 | 0.885889 | 0.963333 | reused |
| B2 | 0.766586 | 0.885889 | 0.963333 | reused |
| B3 | 0.766586 | 0.885889 | 0.963333 | 4504.4 |
| B4 | 0.766586 | 0.885889 | 0.963333 | 4503.2 |
| B5 | 0.766586 | 0.885889 | 0.963333 | reused |

Paired primary-metric differences versus Hybrid; W/T/L counts queries.

| Variant | Metric | Mean Δ | Paired 95% CI | W/T/L |
| --- | --- | --- | --- | --- |
| B0 | ndcg@10 | +0.044947 | [+0.021844, +0.068412] | 62/206/32 |
| B1 | ndcg@10 | +0.044947 | [+0.021844, +0.068412] | 62/206/32 |
| B2 | ndcg@10 | +0.044947 | [+0.021844, +0.068412] | 62/206/32 |
| B3 | ndcg@10 | +0.044947 | [+0.021844, +0.068412] | 62/206/32 |
| B4 | ndcg@10 | +0.044947 | [+0.021844, +0.068412] | 62/206/32 |
| B5 | ndcg@10 | +0.044947 | [+0.021844, +0.068412] | 62/206/32 |

**bright-stackoverflow quality**

| Variant | ndcg@10 | recall@10 | recall@100 | alpha_ndcg@10 | aspect_recall@10 | Mean reranker ms/query |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid | 0.480653 | 0.525183 | 0.793428 | 0.489142 | 0.611014 | — |
| B0 | 0.341227 | 0.415989 | 0.793428 | 0.348156 | 0.487482 | 4637.5 |
| B1 | 0.398871 | 0.462946 | 0.793428 | 0.406470 | 0.532602 | reused |
| B2 | 0.398871 | 0.462946 | 0.793428 | 0.406470 | 0.532602 | reused |
| B3 | 0.407530 | 0.468287 | 0.793428 | 0.412765 | 0.536285 | 4589.0 |
| B4 | 0.388098 | 0.456548 | 0.793428 | 0.394181 | 0.528397 | 4566.1 |
| B5 | 0.407530 | 0.468287 | 0.793428 | 0.412765 | 0.536285 | reused |

Paired primary-metric differences versus Hybrid; W/T/L counts queries.

| Variant | Metric | Mean Δ | Paired 95% CI | W/T/L |
| --- | --- | --- | --- | --- |
| B0 | ndcg@10 | -0.139425 | [-0.184483, -0.094246] | 26/17/72 |
| B0 | aspect_recall@10 | -0.123532 | [-0.174282, -0.074840] | 13/55/47 |
| B1 | ndcg@10 | -0.081781 | [-0.118923, -0.043602] | 22/41/52 |
| B1 | aspect_recall@10 | -0.078411 | [-0.121377, -0.036362] | 12/70/33 |
| B2 | ndcg@10 | -0.081781 | [-0.118923, -0.043602] | 22/41/52 |
| B2 | aspect_recall@10 | -0.078411 | [-0.121377, -0.036362] | 12/70/33 |
| B3 | ndcg@10 | -0.073123 | [-0.116507, -0.029584] | 33/21/61 |
| B3 | aspect_recall@10 | -0.074729 | [-0.119471, -0.031203] | 19/58/38 |
| B4 | ndcg@10 | -0.092554 | [-0.133190, -0.051887] | 30/19/66 |
| B4 | aspect_recall@10 | -0.082616 | [-0.128013, -0.038488] | 18/56/41 |
| B5 | ndcg@10 | -0.073123 | [-0.116507, -0.029584] | 33/21/61 |
| B5 | aspect_recall@10 | -0.074729 | [-0.119471, -0.031203] | 19/58/38 |

**bright-robotics quality**

| Variant | ndcg@10 | recall@10 | recall@100 | alpha_ndcg@10 | aspect_recall@10 | Mean reranker ms/query |
| --- | --- | --- | --- | --- | --- | --- |
| Hybrid | 0.390414 | 0.376942 | 0.649205 | 0.415825 | 0.485526 | — |
| B0 | 0.291856 | 0.311545 | 0.649205 | 0.320568 | 0.437296 | 4610.9 |
| B1 | 0.348986 | 0.353714 | 0.649205 | 0.382271 | 0.481195 | reused |
| B2 | 0.348986 | 0.353714 | 0.649205 | 0.382271 | 0.481195 | reused |
| B3 | 0.355810 | 0.363203 | 0.649205 | 0.387762 | 0.490528 | 4571.7 |
| B4 | 0.340933 | 0.346992 | 0.649205 | 0.372474 | 0.475951 | 4576.2 |
| B5 | 0.355810 | 0.363203 | 0.649205 | 0.387762 | 0.490528 | reused |

Paired primary-metric differences versus Hybrid; W/T/L counts queries.

| Variant | Metric | Mean Δ | Paired 95% CI | W/T/L |
| --- | --- | --- | --- | --- |
| B0 | ndcg@10 | -0.098558 | [-0.140504, -0.058509] | 27/11/63 |
| B0 | aspect_recall@10 | -0.048230 | [-0.096984, -0.002416] | 21/48/32 |
| B1 | ndcg@10 | -0.041428 | [-0.069236, -0.014506] | 21/37/43 |
| B1 | aspect_recall@10 | -0.004332 | [-0.039787, +0.030801] | 17/66/18 |
| B2 | ndcg@10 | -0.041428 | [-0.069236, -0.014506] | 21/37/43 |
| B2 | aspect_recall@10 | -0.004332 | [-0.039787, +0.030801] | 17/66/18 |
| B3 | ndcg@10 | -0.034603 | [-0.065992, -0.003076] | 34/16/51 |
| B3 | aspect_recall@10 | +0.005002 | [-0.034711, +0.043665] | 22/59/20 |
| B4 | ndcg@10 | -0.049481 | [-0.079464, -0.019269] | 35/14/52 |
| B4 | aspect_recall@10 | -0.009575 | [-0.051002, +0.029969] | 21/56/24 |
| B5 | ndcg@10 | -0.034603 | [-0.065992, -0.003076] | 34/16/51 |
| B5 | aspect_recall@10 | +0.005002 | [-0.034711, +0.043665] | 22/59/20 |

**Controlled primary-metric contrasts**

| Contrast | Dataset | Metric | Mean Δ | Paired 95% CI | W/T/L |
| --- | --- | --- | --- | --- | --- |
| B1_versus_B0 | scifact | ndcg@10 | +0.000000 | [+0.000000, +0.000000] | 0/300/0 |
| B1_versus_B0 | bright-stackoverflow | ndcg@10 | +0.057644 | [+0.029379, +0.090823] | 20/91/4 |
| B1_versus_B0 | bright-stackoverflow | aspect_recall@10 | +0.045121 | [+0.016884, +0.078141] | 14/100/1 |
| B1_versus_B0 | bright-robotics | ndcg@10 | +0.057130 | [+0.026412, +0.092654] | 23/71/7 |
| B1_versus_B0 | bright-robotics | aspect_recall@10 | +0.043899 | [+0.014112, +0.078884] | 14/83/4 |
| B3_versus_B2 | scifact | ndcg@10 | +0.000000 | [+0.000000, +0.000000] | 0/300/0 |
| B3_versus_B2 | bright-stackoverflow | ndcg@10 | +0.008658 | [-0.022431, +0.039407] | 36/44/35 |
| B3_versus_B2 | bright-stackoverflow | aspect_recall@10 | +0.003682 | [-0.029362, +0.036120] | 19/81/15 |
| B3_versus_B2 | bright-robotics | ndcg@10 | +0.006825 | [-0.013876, +0.027574] | 31/47/23 |
| B3_versus_B2 | bright-robotics | aspect_recall@10 | +0.009334 | [-0.018412, +0.037711] | 12/79/10 |
| B4_versus_B2 | scifact | ndcg@10 | +0.000000 | [+0.000000, +0.000000] | 0/300/0 |
| B4_versus_B2 | bright-stackoverflow | ndcg@10 | -0.010773 | [-0.038671, +0.016038] | 26/61/28 |
| B4_versus_B2 | bright-stackoverflow | aspect_recall@10 | -0.004205 | [-0.032781, +0.022766] | 14/87/14 |
| B4_versus_B2 | bright-robotics | ndcg@10 | -0.008053 | [-0.027234, +0.009606] | 26/57/18 |
| B4_versus_B2 | bright-robotics | aspect_recall@10 | -0.005243 | [-0.029647, +0.017643] | 8/84/9 |
| B4_versus_B3 | scifact | ndcg@10 | +0.000000 | [+0.000000, +0.000000] | 0/300/0 |
| B4_versus_B3 | bright-stackoverflow | ndcg@10 | -0.019431 | [-0.036728, -0.002957] | 25/51/39 |
| B4_versus_B3 | bright-stackoverflow | aspect_recall@10 | -0.007888 | [-0.029671, +0.012879] | 8/97/10 |
| B4_versus_B3 | bright-robotics | ndcg@10 | -0.014877 | [-0.028085, -0.002949] | 19/56/26 |
| B4_versus_B3 | bright-robotics | aspect_recall@10 | -0.014577 | [-0.033451, +0.003492] | 4/89/8 |

B2 versus B1 is exactly zero for every query and metric. Full secondary-metric intervals are in the checksummed JSON analyses.

**Input diagnostics** — B0/B1 use the same inputs as B2; B5 uses the selected allocation.

| Dataset | Variant | Truncated % | Zero body % | Body <32/64/128 % | Body mean | Body p10/p50/p90 |
| --- | --- | --- | --- | --- | --- | --- |
| scifact | B2 | 20.62 | 0.00 | 0.07/0.33/4.45 | 286.47 | 167.0/289.0/398.0 |
| scifact | B3 | 20.62 | 0.00 | 0.07/0.33/4.45 | 286.47 | 167.0/289.0/398.0 |
| scifact | B4 | 20.62 | 0.00 | 0.07/0.33/4.45 | 286.47 | 167.0/289.0/398.0 |
| bright-stackoverflow | B2 | 68.78 | 26.09 | 34.04/39.96/55.96 | 120.84 | 0.0/106.0/303.0 |
| bright-stackoverflow | B3 | 89.61 | 0.00 | 6.70/11.13/24.22 | 220.62 | 53.0/262.5/313.0 |
| bright-stackoverflow | B4 | 80.48 | 0.00 | 6.70/11.13/24.22 | 197.54 | 53.0/247.0/303.0 |
| bright-robotics | B2 | 56.44 | 25.74 | 34.01/43.27/63.47 | 113.18 | 0.0/85.0/291.0 |
| bright-robotics | B3 | 76.24 | 0.00 | 8.96/18.61/36.63 | 198.27 | 35.0/207.0/313.0 |
| bright-robotics | B4 | 67.72 | 0.00 | 8.96/18.61/36.63 | 178.52 | 35.0/207.0/291.0 |

**Score/rank diagnostics** — promotions/demotions count known-positive documents within the fixed pool, and may occur together in one query.

| Dataset | Variant | Tie queries | All equal | Mean unique scores | Mean rank displacement | Positive ↑/↓ | Fallback % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scifact | B0 | 0 | 0 | 20.000 | 4.078 | 74/41 | 0.00 |
| scifact | B1 | 0 | 0 | 20.000 | 4.078 | 74/41 | 0.00 |
| scifact | B2 | 0 | 0 | 20.000 | 4.078 | 74/41 | 0.00 |
| scifact | B3 | 0 | 0 | 20.000 | 4.078 | 74/41 | 0.00 |
| scifact | B4 | 0 | 0 | 20.000 | 4.078 | 74/41 | 0.00 |
| scifact | B5 | 0 | 0 | 20.000 | 4.078 | 74/41 | 0.00 |
| bright-stackoverflow | B0 | 67 | 30 | 14.113 | 5.644 | 104/197 | 0.00 |
| bright-stackoverflow | B1 | 67 | 30 | 14.113 | 3.861 | 87/141 | 0.00 |
| bright-stackoverflow | B2 | 67 | 30 | 14.113 | 3.861 | 87/141 | 0.00 |
| bright-stackoverflow | B3 | 52 | 0 | 18.835 | 5.138 | 112/176 | 0.00 |
| bright-stackoverflow | B4 | 52 | 0 | 18.835 | 5.176 | 112/182 | 0.00 |
| bright-stackoverflow | B5 | 52 | 0 | 18.835 | 5.138 | 112/176 | 0.00 |
| bright-robotics | B0 | 67 | 26 | 14.317 | 5.619 | 71/155 | 0.00 |
| bright-robotics | B1 | 67 | 26 | 14.317 | 3.895 | 56/107 | 0.00 |
| bright-robotics | B2 | 67 | 26 | 14.317 | 3.895 | 56/107 | 0.00 |
| bright-robotics | B3 | 54 | 0 | 19.030 | 5.246 | 91/124 | 0.00 |
| bright-robotics | B4 | 54 | 0 | 19.030 | 5.237 | 87/129 | 0.00 |
| bright-robotics | B5 | 54 | 0 | 19.030 | 5.246 | 91/124 | 0.00 |

**Fresh CPU reranker latency** — milliseconds; per-candidate values are query time divided by 20. Model loading and upstream retrieval are excluded.

| Dataset | Variant | Query mean | Query p10/p50/p90 | Candidate mean |
| --- | --- | --- | --- | --- |
| scifact | B0 | 4500.9 | 4271.7/4530.9/4705.0 | 225.0 |
| scifact | B3 | 4504.4 | 4286.8/4534.4/4713.4 | 225.2 |
| scifact | B4 | 4503.2 | 4276.7/4532.4/4728.4 | 225.2 |
| bright-stackoverflow | B0 | 4637.5 | 4467.0/4633.0/4801.5 | 231.9 |
| bright-stackoverflow | B3 | 4589.0 | 4382.1/4603.0/4796.6 | 229.5 |
| bright-stackoverflow | B4 | 4566.1 | 4407.2/4571.7/4722.1 | 228.3 |
| bright-robotics | B0 | 4610.9 | 4404.8/4620.8/4811.4 | 230.5 |
| bright-robotics | B3 | 4571.7 | 4267.9/4588.5/4800.0 | 228.6 |
| bright-robotics | B4 | 4576.2 | 4322.7/4603.1/4779.3 | 228.8 |


## 5. Selected shared policy

The preregistered rule selects **B3**, one configuration across all three datasets.
Both allocations pass the SciFact guard. B3 is better than B4 on both nDCG@10
and weighted aspect recall in each technical domain; it therefore wins by the
registered Pareto rule, without using the tie-break or selecting per dataset.
B3–B2 nDCG changes are only +0.008658 on Stack Overflow and +0.006825 on Robotics;
both paired intervals include zero. This is not evidence of a reliable quality
improvement from token allocation alone. B4 is worse than B3 on nDCG in both
domains, with paired intervals excluding zero, but this joint query/body tradeoff
does not isolate which discarded or retained text was decisive.

The optional production scorer now uses:

- The same pinned model, 512 total tokens and batch size 16.
- Query cap **128**, counting the head/tail gap; title head cap **64**.
- Remaining capacity for body, guaranteeing `min(full body tokens, 128)`.
- Stable exact-score ties; unchanged depth **20** in the existing default composition.
- Query-level fallback for all-empty body inputs before model execution;
  exact-all-equal scores for a pool larger than one; unusable scores; or expected
  model execution errors. Fallback preserves original identities, order, methods
  and scores, and records its reason. Equal model scores remain available as
  diagnostic metadata. A single valid candidate retains its model score.

There is no per-candidate removal, score blending, epsilon-variance threshold,
domain branch or Agent routing change. Invalid query/budget configuration and
programming errors still propagate. An impossibly small custom query cap cannot
silently remove every complete query character; this validation affects none of
the registered 128/192-token inputs. Explicit `query_cap=None` remains available
for legacy-input diagnosis, with the corrected ranking/fallback invariants.

B5 runs the final production ordering over all selected real-model scores and
checks every prepared input against B3: **10,320 exact input records and 516 exact
rankings**. Re-scoring the first registered query in each dataset reproduces all
60 scores exactly (maximum error zero), confirming that preparing the whole
query before retaining the original 16/4 scoring batches does not change scoring.
This is a final production replay of the already completed full B3 inference,
not another quality trial. B5 latency fields reuse B3; replay overhead is separate.

B5a's exact-all-equal trigger activates **0/516** times. B5b's all-body-empty
trigger also occurs **0/516** times, so no additional B5b quality arm is warranted.
Both fallback behaviors are covered by deterministic tests; no benchmark gain
is attributed to a fallback that never activated. B6 is **not run**: the
predeclared technical-domain improvement gate fails. Depth remains 20.

After complete aggregate analysis, an input-collision diagnostic examined the
remaining score ties. B3 and B4 each have 177 tied pairs on Stack Overflow and
119 on Robotics; every pair has identical effective inputs. Of those, 163/103
pairs already have identical full candidate title/body text. The remaining
14/16 pairs have different full candidate text but identical visible prefixes
after allocation. No distinct-input exact-score ties were found. These findings
explain why stable ties still matter and identify representation limits without
justifying another threshold or a post-hoc allocation search.

## 6. Hybrid versus corrected reranker

This is the direct product decision comparison. B5 is identical to B3 on every
ranking and metric. Intervals below are paired query bootstrap 95% intervals;
W/T/L counts queries.

| Dataset | Hybrid nDCG@10 | Corrected | Mean Δ | 95% CI | W/T/L | Added reranker time/query |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| SciFact | 0.721640 | 0.766586 | +0.044947 | [+0.021844, +0.068412] | 62/206/32 | 4.50 s |
| Stack Overflow | 0.480653 | 0.407530 | −0.073123 | [−0.116507, −0.029584] | 33/21/61 | 4.59 s |
| Robotics | 0.390414 | 0.355810 | −0.034603 | [−0.065992, −0.003076] | 34/16/51 | 4.57 s |

| Dataset | Hybrid weighted aspect recall@10 | Corrected | Mean Δ | 95% CI | W/T/L |
| --- | ---: | ---: | ---: | --- | --- |
| Stack Overflow | 0.611014 | 0.536285 | −0.074729 | [−0.119471, −0.031203] | 19/58/38 |
| Robotics | 0.485526 | 0.490528 | +0.005002 | [−0.034711, +0.043665] | 22/59/20 |

The extra time is the measured reranker stage over the frozen pool, not a
complete end-to-end Hybrid serving measurement. Recall@100 is unchanged in every
arm. The SciFact gain survives; both technical-domain nDCG regressions have
negative paired upper bounds. Robotics' small aspect-coverage increase remains
uncertain and does not reverse the ranking-quality decision.

Examples were selected only after full aggregate analysis: the two worst
nDCG regressions against Hybrid in each technical domain, with query ID as the
tie-break. The complete queries, lost positives, body counts and categories are
in `post-hoc-examples.json`; they are debugging cases, not prevalence estimates.

| Dataset / query | Query topic | Hybrid → corrected nDCG | Example known-positive demotion | Body retained |
| --- | --- | ---: | --- | ---: |
| Stack Overflow 79 | Structured JSON responses | 0.921787 → 0.156426 | `stackoverflow-79/extraction_0.txt`: 2 → 11 | 312/504 tokens |
| Stack Overflow 3 | Merging a list of DataFrames | 0.753698 → 0.151020 | `stackoverflow-3/extraction_1.txt`: 4 → 15 | 312/443 tokens |
| Robotics 22 | Supplying motors/controllers from one power source | 0.678504 → 0.133052 | `robotics-22/extraction_4.txt`: 2 → 14 | 313/377 tokens |
| Robotics 99 | Gazebo plugin library search path | 0.429694 → 0.000000 | `robotics-99/extraction_0.txt`: 3 → 15 | 312/509 tokens |

These positives were present inside the frozen rerank pool and had substantial
visible body text. Their demotion is observed directly; these examples do not
prove that every necessary query constraint or supporting passage survived.
The model/input-policy combination can still rank useful evidence poorly even
after eliminating completely missing bodies.

## 7. Product recommendation

**Reranking should remain optional.**

The corrected stage has useful scientific-query behavior and stronger execution
invariants, but it still systematically reduces technical-domain nDCG while
adding about 4.5–4.6 seconds per 20-candidate CPU query. Stack Overflow also loses
weighted aspect coverage. This does not meet the preregistered quality gate for
a global default. Token/input reliability is improved; broadly reliable quality
improvement over Hybrid has **not** been established.

Existing opt-in behavior is preserved. The global retrieval mode remains its
existing semantic default; explicit Hybrid retrieval retains Hybrid when
reranking is not requested. Agent model, prompt, tool schemas, evidence protocol,
finalization, budgets and routing remain untouched. This decision is complete
for Phase B and does not depend on further tuning or Phase C.

## 8. Tests

| Test group | Passed | Failed / errors | Skipped |
| --- | ---: | ---: | ---: |
| Deterministic suite | 1152 | 0 | 0 |
| Service integration | 60 | 0 | 0 |
| Evaluation regression subset | 248 | 0 | 0 |
| Phase A regression subset | 378 | 0 | 0 |
| Reranker focused deterministic subset | 48 | 0 | 0 |
| New reranker deterministic cases | 38 | 0 | 0 |
| Reranker real-model integration subset | 5 | 0 | 0 |
| New real long-query integration case | 1 | 0 | 0 |
| Live freshness checks (separate fixture) | 16 | 0 | 0 |

Counts are taken from saved JUnit results; evaluation, Phase A and reranker rows
are subsets of the deterministic suite and must not be added to its total.
Phase A's 378-case regression subset covers Agent (187), exact retrieval (23),
document loading/access (13), Runtime (24), and CLI (131).

The earlier private tokenizer-options assertion was replaced by public prepared
input tests and inspection of the actual tokens sent to an external model stub.
Tests cover stable groups, identical/near-identical scores, one candidate,
distinct chunks of one source, invalid model output, expected execution errors,
programming-error propagation, original evidence/score preservation, field caps,
Unicode boundaries, fixed overhead, short inputs, all-empty query fallback and
an empty first batch with a nonempty later batch. The real-model long technical
query regression additionally checks head/tail retention, body evidence and
relevant-result promotion. The service suite exercises real cached models,
Ollama, Qdrant, snapshot retrieval, filtering and Phase A execution behavior.

Final production source is committed as `75ba733`. A scope audit against
`c47a015` finds only `retrieval/rerank.py`, `retrieval/qwen_rerank.py`, and
`retrieval/qwen_inputs.py` changed under `src/`. No upstream or Agent source
changes are included. The 16 live freshness checks use a separate fixture and
new index publications; they verify edits, deletions, renames, stale references
and both retained remote snapshots without altering accepted P4 snapshots.

## 9. Remaining issues and evidence

**Reranker-model/input limitations.** Full body starvation is fixed, but the
current 0.6B model with this fixed input policy still demotes known positives in
both technical domains. Head/tail selection can omit middle constraints and
body prefixes can omit decisive later evidence. Selected chunks from distinct
sources sometimes duplicate the same text. These limitations are not repaired
by stable ties or a nonempty-body guarantee. No model replacement, summarizer,
learned threshold or additional allocation search was performed.

**Hybrid/fusion limitations.** At least one known positive is absent from Hybrid
100 for 11/300 SciFact, 56/115 Stack Overflow and 77/101 Robotics queries; a known
positive lies in positions 21–100 for 25/300, 52/115 and 61/101 queries. These
categories overlap and do not mean that all positives are missing. A fixed
20-source reranker cannot recover those candidates. The highest Hybrid-ranked
chunk may not contain all relevant source evidence. Fusion, candidate generation,
chunking and pre-pool source aggregation are unchanged.

**Agent-policy limitations.** Phase B evaluates the downstream ranking stage,
not answer correctness, multi-hop planning or tool-selection policy. A future
Agent study would require its own controlled evidence; no benchmark/domain
routing, Agent behavior change or Phase C implementation is included here.

**Evaluation limitations.** These are exposed public development queries,
including a public SciFact test split; they are not independent ARKB release
validation. Metrics inherit P4's supplied qrels/aspects and treatment of unjudged
sources. Source aliases, incomplete judgments and actual answer sufficiency can
require separate review; no gold labels were edited after observing scores.
The preserved scorer environment is one CPU/float32 implementation, fixed batch
layout and checkpoint. Confidence intervals are exploratory and unadjusted for
multiple comparisons; latency is not a dedicated serving/SLA benchmark. The
older SciFact weight/environment provenance gap remains explicit despite exact
score reproduction.

The versioned [Phase B evidence bundle](../evaluation/phase-b/v1/README.md)
contains frozen pools, labels in a separate scoring directory, tokenizer data,
model/dataset hashes, original and measured source, all B0–B5 scores/rankings,
per-query metrics, attribution and strata, experiment ledgers, test logs,
checksums and replay instructions. Original P4 and intermediate Phase B outputs
are retained. Full metric precision and all secondary-metric paired intervals
are available there; no dataset-level scores are merged.

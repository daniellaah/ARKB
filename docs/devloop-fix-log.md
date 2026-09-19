# Fix-first development log

Started 2026-09-18 UTC after the user chose to fix diagnosed product problems
before any further registered study. Every number here comes from the
[fast development loop](../evaluation/devloop/README.md): the product agent
(Qwen3.5-4B, nonthinking, temperature 0, 32,768 context, 4,096 output tokens)
under the study budgets (12/10/6 tool, query, read calls; 8,000 evidence
tokens; 300 s), on 144 development scenarios plus an optional ten-query
long-document slice. These are development measurements on exposed material
with a small model; they support engineering decisions, not quality claims.

## Baseline (pre-fix product, run `baseline-4b-nothink`)

| Slice | n | Primary metric | Value | Elapsed mean |
| --- | ---: | --- | ---: | ---: |
| v2 pilot | 60 | span evidence coverage, delivered | 0.926 | 6.1 s |
| exact-nfcorpus | 24 | completeness of cited sources | 0.722 | 8.4 s |
| recall-nfcorpus | 20 | positive-document recall, delivered | 0.147 | 10.6 s |
| recall-fiqa | 20 | positive-document recall, delivered | 0.311 | 10.6 s |
| musique | 20 | answer F1 | 0.073 | 11.1 s |
| long-browsecomp | 10 | positive-document recall, delivered | 0.154 | 20 s |

Observed failure mechanisms:

- **Enumeration truncates silently.** The agent always called `match` with the
  default limit of five occurrences; several occurrences came from one note and
  nothing told it the list was incomplete. Completeness fell with the number of
  matching notes: 1.00 for 2 to 5 notes, 0.59 for 6 to 12, 0.20 for 13 to 30.
- **One shallow search.** On FiQA and NFCorpus the agent made 1.1 to 1.25
  searches per question, always five results, mostly semantic. Its delivered
  recall (0.31 FiQA, 0.15 NFCorpus) equals what the engine returns at five
  results (0.29 and 0.13 semantic). The engine reaches 0.41 and 0.17 at ten and
  0.51 and 0.20 at twenty results, so depth and reformulation, not ranking,
  bound the agent here. NFCorpus averages 31 positives per query, so recall
  over all positives is inherently low for any five-result call.
- **Evidence budget spent on packing, not reading.** On BrowseComp-Plus, nine
  of ten trajectories stopped at the 8,000-token evidence ceiling after two or
  three searches, because every search delivered five to ten whole 512-token
  chunks (2,200 to 4,800 tokens per call). Reads were rare; one whole-document
  read returned 248,286 tokens and was withheld, which closed the trajectory.
- **Small-model limits, not tooling.** MuSiQue ended with `insufficient_evidence`
  in 19 of 20 trajectories with only about 1,000 evidence tokens used; the v2
  `no_retrieval` tasks were never respected (the model searched for arithmetic).
  These belong to the model-capacity axis, not to the tool layer.

## Fix 1: enumeration completeness (`match`)

Change: `match` accepts `unique_sources` (one hit per note) and every response
reports `truncated` when more matches exist beyond `limit`; the tool description
explains both. Runtime and CLI expose `--unique-sources`. Retrieval and tool
tests cover truncation detection in occurrence and unique modes.

| Run | exact completeness (cited) | all 24 complete and exact | elapsed | v2 answered |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.722 | 11/24 | 8.4 s | 49/60 |
| tool change + prompt line | 1.000 | 24/24 | 17.7 s | 42/60 |
| tool change, schema guidance only | pending | pending | pending | pending |

The first variant added a sentence to the product system prompt telling the
model to use `unique_sources` for enumeration. It reached full completeness on
every stratum (13 wins, 0 losses) but cost more calls and tokens, and it biased
the 4B model toward `match` in semantic tasks: seven v2 questions that had been
answered ended in `insufficient_evidence` after the model burned turns on
repeated `match` calls. The prompt line was removed. With guidance only in the
tool description, the model used `unique_sources` on its first call but with the
default limit of five; when `truncated` came back it retried with a limit of
100, usually without `unique_sources`, which still completed 21 of 24 but cost 45
to 54 s on the largest stratum. Three failures retried with a different term or
without raising the limit. Two further adjustments: an omitted `limit` now means
50 notes whenever `unique_sources` is set (one hit per note costs a few tokens),
and the description was shortened because the longer text drew the 4B model into
`match` calls during semantic tasks (v2 match calls 34 to 45, max-turn endings 9
to 14). The v2 answered count moved 49, 42, 46 across the three runs; the
repeatability study showed two thirds of cells change their final object between
identical runs, so single-run differences of this size on 60 questions are
tracked, not chased.

## Fix 3: bounded reads and partial evidence delivery

Change: `read(ref)` now defaults to `expand=section`, the heading section that
contains the evidence (bounded to a 3,000-character window around `match`
hits); `document` must be requested explicitly and the schema says it can be
very long. When a multi-hit result does not fit the remaining evidence
allowance, the fitting prefix is delivered and the rest is recorded as withheld
with a note; a single oversized hit is withheld with an `evidence_too_large`
error that names the sizes and suggests smaller units, and collection continues
while at least one tenth of the allowance remains. Delivered and withheld hits
are recorded per reference, so evidence accounting stays exact.

Full development set with fixes 1, 1b and 3 (run `packing-fix-4b-nothink`,
paired against the baseline): exact completeness 1.000 (13 wins, 0 losses,
every first call now `unique_sources=true, limit=50`, elapsed 13.5 s versus
8.4 s); v2 span coverage 0.957 (+0.03); recall and MuSiQue unchanged within
noise; no withheld results on these short-document tracks, as expected.

Long-document slice (ten BrowseComp-Plus queries, run
`long-packing-fix-4b-nothink` against `long-baseline-4b-nothink`): the
catastrophic patterns are gone. No whole-document read appears (the baseline
had one of 248,286 tokens that closed its trajectory); three reads used the
bounded section; partial delivery happened 14 times and recorded 81 withheld
hits; the trajectories now use 7,208 of the 8,000 evidence tokens on average
instead of 5,083 and run about six seconds longer. Outcomes barely moved:
delivered positive recall 0.154 to 0.166 (two wins, one loss), every trajectory
still ends in `insufficient_evidence`. On this track the 4B model, not the
packing, is the limit; the fix makes long-document behavior safe and
accountable rather than better at answering.

One tracked side effect: the 4B model now issues more `match` calls inside
semantic tasks (v2 match calls 34 to 54, mostly in semantic discovery), and
several trajectories that used to finish now hit the turn limit; v2 answered
went 49 to 45 of 60. Three post-change runs all sit a few answers below the
baseline, so this is probably a small real cost of a more capable `match` on a
small model rather than noise alone. It is recorded, not chased, until the
model-capacity axis is measured.

## Fix 2: candidate acquisition

Change: the default search depth is ten chunks instead of five (tool schema,
session and adapter defaults), the schema explains `limit` (up to 20 for broad
questions), and the search description tells the model to search again with
different wording, another mode or a larger limit when results do not cover
the question. Partial delivery (fix 3) keeps deeper results affordable.

Measured on the full development set (`acquisition-fix-4b-nothink` against
`packing-fix-4b-nothink`): FiQA delivered recall 0.311 to 0.324 (two wins, one
loss), NFCorpus 0.156 to 0.170 (five wins, no losses), exact and v2 coverage
unchanged, elapsed up 2 to 6 s per question on the recall tracks. The gain is
far below the engine's recall@10 because the 4B model overrides the default:
it wrote `limit=5` explicitly in 10 of 24 FiQA searches and still made only
1.2 to 1.3 searches per question. The deeper default is kept as the product
setting (more recall per call at a modest, accounted cost); the remaining gap
is model behavior, to be re-measured on the capacity axis. On the long-document
slice the deeper default changed nothing (delivered recall 0.166 both ways);
the model now asks for 10 to 20 chunks, partial delivery withholds 121 hits
across ten trajectories, and one trajectory ended in a model output error.

## Status after the three fixes (2026-09-18 UTC)

| Slice | Baseline | After fixes 1, 1b, 2, 3 | Note |
| --- | ---: | ---: | --- |
| exact-nfcorpus completeness (cited) | 0.722 | 1.000 | 24/24 complete and exact; elapsed 8.4 to 16.0 s |
| v2 span coverage (delivered) | 0.926 | 0.947 | answered 49 to 43 of 60; more `match` calls in semantic tasks |
| recall-fiqa (delivered) | 0.311 | 0.324 | engine recall@10 would allow 0.41 |
| recall-nfcorpus (delivered) | 0.147 | 0.170 | 31 positives per query on average |
| musique answer F1 | 0.073 | 0.083 | 19 of 20 abstain; model-capacity limit |
| long-browsecomp recall (delivered) | 0.154 | 0.166 | no whole-document reads; budget used 5.1k to 6.5k tokens |

What the tool layer now guarantees: complete enumeration with an explicit
truncation signal, bounded reads by default, exact accounting of delivered and
withheld evidence, and a deeper default search. What it cannot fix: a 4B
nonthinking model that overrides depth, rarely reformulates, mistakes literal
`match` for concept search, and abstains on multi-hop questions. The next
measurement should be the model-capacity axis (9B and 27B, thinking on) on this
same development set, followed by the product-shaped benchmark.

## Engine-only reference (no model)

| Dataset | Mode | recall@5 | recall@10 | recall@20 |
| --- | --- | ---: | ---: | ---: |
| FiQA | BM25 | 0.118 | 0.186 | 0.308 |
| FiQA | Semantic | 0.294 | 0.407 | 0.505 |
| FiQA | Hybrid | 0.242 | 0.344 | 0.462 |
| NFCorpus | BM25 | 0.124 | 0.144 | 0.157 |
| NFCorpus | Semantic | 0.133 | 0.168 | 0.201 |
| NFCorpus | Hybrid | 0.144 | 0.161 | 0.190 |

Twenty queries per dataset, document-level positives, one call per mode.

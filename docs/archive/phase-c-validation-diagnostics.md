# Phase C supplementary validation diagnostics

The saved FiQA rankings show a substantial Hybrid ranking loss relative to
Semantic, while NFCorpus shows both candidate coverage limitations and an
uncertain small Hybrid improvement. These findings identify questions for future
development; they do not reopen the frozen Phase C selection of C0.

This analysis completed while BrowseComp-Plus embedding continued. It used all
648 FiQA and 323 NFCorpus queries, unchanged qrels, and existing BM25, Semantic
and C0 TREC rankings. It made zero model or retrieval calls. No production
configuration, live index, background script or original evaluation artifact was
changed. BrowseComp-Plus validation remains pending.

## Method and interpretation

This is a **post-hoc descriptive analysis**: the aggregate validation results
were already known. The [analysis protocol](../evaluation/phase-c/v1/validation-diagnostics/protocol.json)
was recorded before calculating the new diagnostics, and is not a preregistration
of an unseen validation experiment.

The paired differences below use 10,000 query bootstrap resamples, seed 20260912,
separately within each dataset. Intervals are exploratory percentile 95%
intervals, without correction for the eight reported dataset/metric comparisons.
Wins, ties and losses use exact per-query metric differences. They do not measure
repeat-run stability or establish generalization to ARKB's native workloads.

The candidate union contains the distinct sources in the captured 500-chunk
BM25 and Semantic legs. It is not exhaustive retrieval over every document.
Positive means an original qrel grade greater than zero. All original relevant
documents remain in denominators, including FiQA's previously documented blank
document exception. Unjudged documents remain unknown. Source relevance alone
cannot determine whether a chosen chunk contains sufficient evidence.

## Hybrid versus Semantic

| Dataset | Metric | C0 minus Semantic | Exploratory 95% interval | Wins / ties / losses |
| --- | --- | ---: | --- | --- |
| FiQA | nDCG@10 | -0.075255 | [-0.095104, -0.055618] | 146 / 241 / 261 |
| FiQA | Recall@10 | -0.082864 | [-0.108004, -0.057299] | 55 / 435 / 158 |
| FiQA | Recall@20 | -0.070410 | [-0.092893, -0.048856] | 45 / 461 / 142 |
| FiQA | Recall@100 | -0.034820 | [-0.050480, -0.019658] | 35 / 536 / 77 |
| NFCorpus | nDCG@10 | +0.008733 | [-0.004258, +0.021903] | 104 / 125 / 94 |
| NFCorpus | Recall@10 | +0.002568 | [-0.004091, +0.009066] | 57 / 208 / 58 |
| NFCorpus | Recall@20 | -0.001473 | [-0.012703, +0.008519] | 71 / 180 / 72 |
| NFCorpus | Recall@100 | +0.002289 | [-0.007902, +0.011964] | 86 / 181 / 56 |

FiQA C0 nDCG@10 is 0.369652 versus Semantic 0.444907. Its loss is not confined
to a few extreme queries: there are 261 losses versus 146 wins, and the interval
is wholly below zero. NFCorpus C0 is 0.362743 versus Semantic 0.354010, but all
four intervals include zero. Its positive average does not establish a robust
improvement.

## Candidate coverage and ranking loss

The following percentages are **macro averages over queries**, each using that
query's complete positive-label denominator. They must not be confused with
pooled document counts or percentages of failed queries.

| Dataset | Relevant coverage in captured union | C0 Recall@100 | Relevant fraction in union but below C0 top 100 | Relevant fraction absent from union |
| --- | ---: | ---: | ---: | ---: |
| FiQA | 91.60% | 74.92% | 16.68% | 8.40% |
| NFCorpus | 56.32% | 33.53% | 22.79% | 43.68% |

For FiQA, 340 of the 1,706 positive query-document pairs are in the union but
below C0 top 100, and 179 are absent from the union. At top 10, the union contains
positive evidence for 208 queries whose C0 top 10 contains no judged positive.
Fourteen queries have no judged positive anywhere in the captured union.

For NFCorpus, 2,793 of the 12,334 positive query-document pairs are in the union
but below C0 top 100, and 6,864 are absent. Sixty-eight queries have a union
positive but none in C0 top 10; 16 have no union positive.

The union-to-cutoff gap is not all achievable by better sorting: some queries
have more positive candidates than available positions. An oracle that orders
all positive candidates first has Recall@100 of 91.60% on FiQA and 56.09% on
NFCorpus. Thus NFCorpus's 22.79-point gap contains 0.23 points of unavoidable
top-100 capacity loss and 22.56 points of ranking headroom. These label-informed
oracles are diagnostic bounds, not realizable performance claims.

## Relevant-source movements

| Dataset | Semantic top-10 positives lost by C0 | C0 top-10 positives added relative to Semantic | BM25-only positives in full captured union |
| --- | --- | --- | --- |
| FiQA | 208 pairs across 173 queries | 79 pairs across 72 queries | 37 pairs across 34 queries |
| NFCorpus | 203 pairs across 108 queries | 203 pairs across 106 queries | 523 pairs across 142 queries |

Lost and added query groups overlap. Counts cannot be subtracted to obtain the
number of queries helped or hurt, and equal counts do not imply equal nDCG:
grades, positions and per-query denominators still matter. BM25-only positives
mean absent from the captured Semantic leg, not inherently unreachable by a
different semantic retriever.

Two FiQA examples illustrate opposite outcomes. Query `1074` has a single
relevant document, `443960`: Semantic ranks it first, BM25 ranks it 150th and
C0 ranks it 17th. Query `5155` has relevant document `462892` at Semantic rank
23, BM25 rank 1 and C0 rank 1. These cases illustrate both ranking loss and
lexical contribution; source rankings alone do not isolate the contribution of
chunk duplication or individual RRF votes.

Examples were selected as the three largest nDCG losses and gains, breaking ties
by query ID. They are not a representative sample. All query IDs, graded positive
document positions and cutoff partitions are preserved in the per-query files.

## Evaluation directions supported by these observations

1. **FiQA: inspect ranking preservation first.** Most judged evidence is already
   somewhere in the captured union. Inspect the saved chunk contributions to
   distinguish high semantic candidates displaced by cross-leg agreement from
   effects of repeated chunks. Any resulting hypothesis must be developed and
   selected on development data, then tested on fresh held-out data.
2. **NFCorpus: investigate candidate coverage as well as sorting.** The 43.68%
   macro positive fraction missing from the union cannot be recovered by
   rearranging those candidates. Increasing depth, changing query formulation or
   changing embeddings would require a separately specified future experiment;
   this analysis tested none of them.
3. **Preserve a test for lexical contribution.** BM25 adds unique judged positives
   on both datasets. FiQA's aggregate Hybrid loss does not justify universally
   disabling BM25 or introducing a validation-specific routing rule.
4. **Separate source and evidence-span evaluation.** These source qrels cannot
   decide whether a representative chunk is useful to an Agent. Future native
   labeling should record evidence spans separately from source relevance.

Current Phase C still retains C0 under its frozen development gate. No C1/C2
retuning, alternative weighting, query rewriting, reranker or Agent experiment
was run on these validation labels.

## Verification and reproduction

The audit verifies every consumed TREC, summary, protocol, manifest and label file
against its recorded checksum, anchors run manifests to the prior successful
full replay records, and checks input hashes again after analysis. It does not
rehash unconsumed large indexes or raw legs in this supplemental run.

All recalculated aggregate metrics exactly match the saved summaries. There are
17,478 independent `pytrec_eval` nDCG/Recall comparisons, with maximum numerical
error 4.45e-16, and 2,913 per-query cutoff partition checks. A hand-computable
mixed-loss fixture also verifies candidate absence, ranking loss and cutoff
capacity; an all-tie fixture checks bootstrap behavior. These are analysis
checks, not additional product regression test cases.

Results: [summary](../evaluation/phase-c/v1/validation-diagnostics/summary.json),
[checksums](../evaluation/phase-c/v1/validation-diagnostics/checksums.json),
[FiQA per-query records](../evaluation/phase-c/v1/validation-diagnostics/fiqa-per-query.jsonl.gz),
[NFCorpus per-query records](../evaluation/phase-c/v1/validation-diagnostics/nfcorpus-per-query.jsonl.gz).

Run from the repository root with a new output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python evaluation/audits/diagnose_phase_c_validation.py \
  --storage /Volumes/ARKBPhaseC \
  --output evaluation/results/phase-c-v1/validation-diagnostics-reproduction
```

The computation is bounded to one process and reads approximately 60 MB of
saved rankings, plus small metadata and labels. It does not use the GPU or
request service work. This is not a controlled measurement of interference with
the concurrent embedding job.

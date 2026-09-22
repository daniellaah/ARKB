# Phase C: architecture audit and ordered protocol

Status: architecture audit recorded before alternative implementation or scoring.
Accepted production baseline: `75ba733`. Starting clean checkout: `84d2735`
(Phase B evidence only). Phase A execution and Phase B B3 reranker are frozen.

## Actual architecture

`Runtime.search` and `AgentTools.search` both delegate to `RetrievalEngine.search`.
The engine constructs `HybridRetriever` with the same BM25/Semantic implementations.
The default per-leg candidate depth is 20; the caller may configure it. A Hybrid
request cannot exceed that depth. Both legs receive the source-equality filter
before their top-k truncation. Both responses must identify the same pinned index.
BM25 uses title + body, unique NFC/casefold word query terms, k1=1.2 and b=.75.
Semantic embeds the original query once and reads the pinned vector index. Neither
leg normalizes its scores for fusion. Both legs have stable identity tie rules.

`rrf` sorts named legs and assigns each distinct chunk identity at most one vote
per leg, with its original one-based rank: sum(1/(60 + rank)). Duplicate identical
hits do not vote twice, but still occupy their input positions. Distinct chunks
of one source are distinct candidates; their votes are **not** added together.
Equal fused scores sort by `(source_id, identity-kind, chunk_id/span)`.
The first leg's verbatim evidence is retained, with per-leg score/rank provenance.
Conflicting identity text, coordinates or known revision/index/vault are rejected.
Production Hybrid truncates fused **chunks** to top_k. It does **not** collapse
sources. Optional `RerankedRetriever` scores the first 20 candidate chunks before
final top_k; B3 applies unchanged. Agent observations select concise evidence
fields, so internal fusion metadata is not sent to the LLM.

P4's external evaluation is a separate adapter: it retrieves 500 chunks per leg,
fuses the whole union, keeps the first occurrence of each `source` and then takes
100 unique sources. The chosen representative is that first fused chunk. Optional
P4 reranking uses its first 20 unique sources, retaining the tail. CLI and Agent
never implicitly invoke this benchmark adapter. C0 means this **accepted source
ranking evaluation projection** of production chunk RRF, not an assertion that
production already returns unique sources. Source addresses are one-to-one with
source IDs in these frozen indexed corpora; that correspondence must be checked.

## Ordered evidence gates

1. Hydrate all saved P4/Phase B leg identities/scores from their immutable SQLite
   snapshots. Save original query text and full SearchResults, separately from
   qrels/aspects. Preserve section, title, text, coordinates and revision. Hash
   query IDs/text, corpus, configuration, both legs and upstream artifacts.
2. Run unchanged production RRF and P4 first-source collapse. Require exact
   agreement for every accepted top100 rank and first20 representative/score.
   Reference nDCG@10: SciFact .7216396110820045, Stack Overflow
   .4806525133795389, Robotics .3904138067602195.
3. Save candidate availability, lexical complementarity, source duplication,
   fusion loss and representative disagreement before defining alternatives.
4. Append a preregistration for C1/C2, representative choice, statistics, decision
   thresholds and conditional C3. Then implement and evaluate frozen-leg variants.
5. Freeze exactly one development selection before integrating/scoring broader
   validation. NFCorpus, FiQA and BrowseComp-Plus use full intended corpora;
   any necessary query sample must be registered independently of outcomes.
6. No validation-driven tuning. Complete regressions, live freshness checks,
   scope audit and reproducible archive before the final 12-section report.

Primary reranking is disabled. No model, Agent, chunking, tokenizer, identity,
index-publication or freshness changes are authorized by this study. No new
benchmark-specific production path. Public development results are not release
claims about private knowledge bases. Test seams are those explicitly requested
in the Phase C specification: source-fusion SearchResult/SearchResponse behavior,
frozen evaluation replay, external data normalization and existing live/index
interfaces. Gold is only available to scoring/diagnostics, never fusion inputs.

## Preregistered alternatives (after C0 diagnostics, before implementation)

C0 reproduced 51,600 source positions and 10,320 representatives/scores across
516 queries. Union-positive-loss query counts: SciFact 10, Stack Overflow 49,
Robotics 60. Mean union recall: .996667, .940890, .853976; C0 Recall@100:
.963333, .793428, .649205. These non-exclusive counts use any missing positive.

All policies consume identical 500-chunk legs; k=60, source top100, no reranker.
New policies visit leg names in sorted order and chunks in input order. Exact
source/representative ties preserve first encounter, independent of document ID.
C0 retains its historical identity tie rule for exact reproduction.

**C1:** Within each leg, keep the first distinct chunk of each source, retaining
raw chunk rank for provenance. Assign compact one-based source ranks in this
first-occurrence order. Between legs, source score is sum(1/(60+source_rank)).
Only that leg's best chunk explains its vote.

**C2:** Within each leg, retain at most the first two distinct chunks per source.
Compute A_leg(source)=sum(1/(60+raw_chunk_rank)) over those chunks. Sort sources
by descending A, stable first-encounter ties, assigning compact source ranks.
Between legs use exactly the C1 source RRF formula. Thus raw ranks appear only
in within-leg aggregation; compact ranks appear in between-leg fusion. If N
were 1 this *two-stage* definition is mathematically equivalent to C1, unlike
directly summing raw chunk contributions across legs. N is fixed at 2; no sweep.
Duplicate identical chunk entries cannot vote twice but retain input positions.

**R0 representative:** Attribute each final per-leg source vote among its retained
chunks in proportion to their within-leg contributions (C1: all to its one chunk).
Sum attributed contributions for the same chunk across legs, then select the
largest, preserving first encounter for exact ties. Return the original content,
coordinates, section and revision unchanged. C0 keeps its fusion-best chunk.
No R1 experiment: source qrels cannot establish chunk-level evidence superiority.
Report representative disagreement and exact span/provenance validity only.

**Conditional C3 gate:** C2 must lose at least .01 mean nDCG@10 relative to C1 on
a BRIGHT domain, AND at least 10% of that domain's queries must contain a new
C2 top10 nonpositive source (relative to C1) explained by two highly repetitive
chunks in at least one leg (whitespace-token set Jaccard >=.90), with a C1 top10
positive lost. Labels are used only for this diagnostic. If both conditions hold,
run exactly one C3: cap each A_leg at 1.5 times its best raw-rank contribution,
then compact-rank/fuse as C2; allocate its final vote proportionally as in C2.
Otherwise skip C3. This gate targets measured redundant-chunk domination rather
than assuming all multi-chunk support is harmful.

**Statistics/cost:** Per dataset, paired bootstrap of query deltas, NumPy
seed=20260911, 10,000 resamples; exploratory percentile 95% intervals and exact
wins/ties/losses. Report nDCG@10, Recall@10/20/100, MRR@10 and both BRIGHT aspect
metrics. No pooled score or independence claim. Time only CPU fusion + result
construction + source collapse, excluding input decoding and all retrieval/model
calls. One warmup per policy; three timed repetitions per query with rotated
policy order; report query-average mean/p50/p95. Preserve all raw timings.

**Development choice:** A policy is eligible only if SciFact nDCG@10 delta >=-.01,
both BRIGHT nDCG@10 deltas >=-.005, every dataset Recall@10/20/100 delta >=-.01,
and both BRIGHT aspect-recall/alpha-nDCG deltas >=-.01. Require at least one BRIGHT
nDCG@10 gain >=.01 with paired CI lower bound >0. Prefer C1 if eligible, then C2,
then gated C3; otherwise retain C0. Exactly one shared selection; no domain routing.

**Broader validation:** Freeze that choice before acquisition/integration runs.
Use complete BEIR NFCorpus test (323), FiQA test (648) and complete original
BrowseComp-Plus queries (830), subject to verified official counts. Full intended
corpora. No query subsampling is currently planned. Compare BM25, Semantic, C0,
and selected policy (reuse C0 if selected). Keep leg depth 500. For BrowseComp's
official Recall@5/100/1000 and nDCG@10, serialize the available full source order
(up to 1000 union sources); the primary ranking remains its first100. Report both
evidence and gold judgments separately and explicitly report actual returned
depths, which may be below1000. Do not imply a 1000-chunk leg retrieval. No answers
or support labels enter queries or ranking. Reject adoption if any broader dataset
has nDCG@10 or Recall@100 delta < -.03 versus C0 (evidence labels primary for
BrowseComp). Other broader results are descriptive; no return to policy tuning.

Official references checked September 11, 2026: [BEIR](https://github.com/beir-cellar/beir),
[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus), and its
[dataset card](https://huggingface.co/datasets/Tevatron/browsecomp-plus).
The benchmark distinguishes evidence documents from answer-containing gold
support; neither is equivalent to generated-answer accuracy.

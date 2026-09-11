# Phase B audit and execution plan

## Frozen boundary (before behavior changes)

Starting tree: clean, `00e9bb7`; accepted Phase A production implementation:
`c47a015`. The later commit adds evaluation evidence only. Phase B changes no
Agent contract, prompt, lifecycle, budget, retrieval leg, fusion, index, or model.

P4 `run_p4.py` obtains 500 chunk hits from each leg, performs existing RRF
(`k=60`), then selects the first occurrence of each source. Its rerank pool is
the first 20 unique sources, represented by each source's highest Hybrid-ranked
chunk. This is **not necessarily chunk index zero**. Sources 21–100 are appended
unchanged after reranking. Full selected SearchResults and all reranker scores
are saved. No duplicate source enters that P4 pool.

Production `RerankedRetriever` is more general: it reranks the input retriever's
chunk candidates, then applies `top_k`. It does not perform P4's source collapse
or append a 100-source tail. Distinct chunks of one source can therefore enter
that production pool; duplicate SearchResult identities are rejected. Phase B
must preserve both existing boundaries rather than move source aggregation.

`QwenRerankerScorer` pins Qwen3-Reranker-0.6B revision
`e61197ed45024b0ed8a2d74b80b4d909f1255473`, CPU float32, SDPA, batch size 16.
It constructs `<Instruct>`, query, `<Document>`, title, two newlines and chunk
body as **one sequence**. The tokenizer's right truncation retains the beginning
of that sequence after reserving the fixed chat prefix/suffix. Thus query,
title and body can each lose their ends; sufficiently long queries remove the
document label, title and body entirely. Left padding preserves the final
scoring position. The deployed budget is 512, not the checkpoint's maximum.

Scores are final-position `logit(yes) - logit(no)`, computed in float32 and
converted to Python floats without probability conversion or rounding. Equality
means exact numerical equality. `Reranker` currently sorts by negative score
and then document/chunk identity, so ties overwrite Hybrid order. It raises
on unusable scores or scorer errors; there is no deterministic fallback.
No dataset-dependent scoring logic exists in the production reranker.

The pinned [official model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/e61197ed45024b0ed8a2d74b80b4d909f1255473/README.md)
confirms the instruction/chat wrapper and reserved-suffix scoring recipe.
We keep that wrapper, model, precision, total length and batching fixed.

## Sequence and gates

1. Freeze all 300 SciFact, 115 Stack Overflow and 101 Robotics P4 queries and
   candidate pools. Validate source text, coordinates, identities, saved ranks,
   dataset hashes and P4 checksums. Store qrels/aspects separately from inputs.
2. Replay saved B0 scores exactly, then run the unchanged scorer on every frozen
   input. Require rank/metric reproduction; compare scores with absolute
   tolerance `1e-4`. Preserve and investigate deviations before quality changes.
3. B1: stable ties only, using saved B0 scores without additional inference.
   Add public-boundary deterministic ordering/preservation tests first.
4. B2: instrument unchanged input construction; require identical token IDs and
   B1 rankings. Inspect query structure without relevance labels, measure exact
   wrapper costs, and predeclare the small allocation matrix before inference.
5. B3/B4: two fixed caps at the existing total length; run every registered
   query once per variant. Preserve raw input IDs, scores, ranks and latency.
6. B5: only justified exact-equality/all-empty/error fallbacks, replaying saved
   scores where possible. Select one general policy by the predeclared rule;
   do not choose separately by dataset. B6 depth is optional, not a prerequisite.
7. Independently rescore all rankings, validate candidate/tail preservation,
   and report paired bootstrap intervals, W/T/L, source/aspect attribution and
   diagnostic strata. Do not merge datasets or substitute selected examples for
   aggregate results. Report negative quality findings without tuning them away.
8. Run deterministic, evaluation, Phase A, real service and applicable freshness
   regressions; audit scope; deliver a report and checksummed evidence.

Test seams authorized by the Phase B request: `Reranker.rerank`,
`RerankedRetriever.search`, measurable Qwen input construction/scoring, existing
Runtime/Agent integration, and scoring-side frozen-artifact replay. Expected
model execution failures may preserve incoming ranks with explicit diagnostics;
input/programming errors must not be hidden by an indiscriminate catch.

All experiments are exposed public development evidence. No release-level
generalization claim, model replacement, Agent changes, or Phase C work.

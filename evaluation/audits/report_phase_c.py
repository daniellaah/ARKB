"""Render the Phase C report from recorded evidence; missing results stay explicit."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
V=ROOT/'evaluation/phase-c/v1'
NAMES={'scifact':'SciFact','bright-stackoverflow':'BRIGHT Stack Overflow','bright-robotics':'BRIGHT Robotics',
       'nfcorpus':'NFCorpus','fiqa':'FiQA','browsecomp-plus':'BrowseComp-Plus'}

def read(name):return json.loads((V/name).read_text())
def fmt(v):return f'{v:.6f}' if isinstance(v,float) else str(v)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(fmt,r))+' |' for r in rows]])+'\n'

def main():
    dev=read('development-summary.json');c0=read('c0-diagnostics.json');extra=read('extra-diagnostics.json')
    decision=read('development-decision.json');parts=[]
    def add(s):parts.append(s.strip()+'\n')
    complete=all((V/f'validation/{ds}-replay.json').exists() for ds in ('nfcorpus','fiqa','browsecomp-plus'))
    finished=(V/'completion.json').exists() and read('completion.json')['status']=='completed'
    tests=read('verification/test-summary.json') if (V/'verification/test-summary.json').exists() else None
    add('# ARKB Phase C: candidate generation and source-level fusion')
    add('Status: '+('**Completed.** All three full validations, offline replays, regression checks and index archives are verified.' if finished else 'validation complete; index preservation is still in progress.' if complete else
        '**In progress.** Development selection is frozen; FiQA/BrowseComp-Plus full validation and final archival are still running. This is not a completed Phase C release.'))
    add('The frozen product decision is **A: keep current chunk-level Hybrid fusion**. C1 and C2 do not satisfy the shared development gate. No runtime fusion, Agent, reranker, embedding, parser, chunking or publication behavior was changed. No later phase is implemented.')
    add('## 1. Baseline reproduction')
    add('Accepted Phase B production baseline: `75ba733`. The clean Phase C starting checkout was `84d2735` (Phase B report/evidence only). Development selection was committed at `5bd06c6`. Actual measured source hashes, rather than commit labels alone, identify every experiment. C0 reproduction preceded alternative implementation. Frozen P4 SQLite snapshots supplied exact chunk content and coordinates missing from earlier compact leg metadata.')
    add(table(['Dataset','Queries','Exact source ranks','Exact first-20 representatives','C0 nDCG@10'],[
        [NAMES[ds],v['queries'],v['exact_source_ranks'],v['exact_representative_checks'],v['metrics']['ndcg@10']] for ds,v in c0.items()]))
    add('All 516 queries, 51,600 top-100 source ranks and 10,320 first-20 representative scores, revisions and spans match the accepted baseline exactly. Aggregate floating-point differences are at most rounding error. The later offline replay verified 258,000 rank positions and representative/provenance records across all five development arms, plus 62 paired-statistic replays, without retrieval or model calls. An initial metadata hydration draft is preserved separately; canonical semantic score type is `cosine_similarity`. It changed no score or ranking. See [baseline diagnostics](../evaluation/phase-c/v1/c0-diagnostics.json) and [offline replay](../evaluation/phase-c/v1/development-replay.json).')
    add('## 2. Current architecture')
    add('Production BM25 and Semantic retrieval return chunks from the same captured index snapshot. BM25 retains k1=1.2, b=0.75, NFC/casefold word tokenization over title and body, and unique query terms. Semantic retrieval retains the original query, pinned Qwen3 embedding model, 1,024 dimensions and cosine scoring. Exact source filters apply upstream. The normal runtime candidate depth remains 20 per leg; this benchmark retains the accepted P4 depth of 500 per leg and exact Qdrant retrieval.')
    add('Production Hybrid applies rank-only RRF with k=60 to chunk identities, one vote per chunk per named leg. Leg names are sorted and ties use the existing chunk identity order. **Production does not collapse sources.** The accepted P4 evaluation projection fuses the complete returned chunk union, keeps the first chunk of each source and then takes 100 sources. That first fused chunk is its representative. CLI and Agent use the same RetrievalEngine; Agent exposes its compact tool contract. This benchmark projection must not be confused with a runtime source-ranking API.')
    add('Primary experiments stop before reranking. Phase B B3 is unchanged. The only source additions/changes are evaluation modules (`source_fusion.py`, the generic BEIR adapter and the separate BrowseComp adapter). The [scope audit](../evaluation/phase-c/v1/scope-audit.json) compares all source files, including newly added ones, against the accepted baseline.')
    add('## 3. Candidate availability')
    add('Recall is the mean of per-query positive-source recall. Leg and union columns use every source found within the frozen top-500 chunk legs; C0 uses its top 100 sources. The gap includes the final cutoff and ordering, so it is not evidence that fusion deleted candidates from its full union.')
    add(table(['Dataset','BM25 full-leg recall','Semantic full-leg recall','Union recall','C0 Recall@100','Union-positive lost queries'],[
        [NAMES[ds],v['leg_recall']['bm25'],v['leg_recall']['semantic'],v['union_recall'],v['metrics']['recall@100'],v['group_query_counts']['union_positive_lost_top100']] for ds,v in c0.items()]))
    add(table(['Dataset','BM25-only positives','Semantic-only positives','Shared positives','Absent-both positives','Lost at C0 top100 positives'],[
        [NAMES[ds],*[v['group_positive_counts'][k] for k in ('bm25_only_positive','semantic_only_positive','shared_positive','positive_absent_both','union_positive_lost_top100')]] for ds,v in c0.items()]))
    add('Positive counts are query–document pairs, not globally distinct documents. BM25 contributes complementary positives in every domain: 1, 34 and 66 respectively. Do not disable it globally. Both candidate generation and final ranking limit BRIGHT: union recall is 0.9409 on Stack Overflow and 0.8540 on Robotics, but C0 top-100 recall is 0.7934 and 0.6492.')
    add('## 4. Source duplication analysis')
    add(table(['Dataset','Unique BM25 chunk-top100','Unique Semantic chunk-top100','Unique fused chunk-top100','Duplicate competition queries'],[
        [NAMES[ds],v['unique_leg_top100']['bm25']['mean'],v['unique_leg_top100']['semantic']['mean'],v['unique_fused_chunks_top100']['mean'],v['duplicate_competition_queries']] for ds,v in c0.items()]))
    add('Every C0/C1/C2 final source list contains 100 distinct sources. Duplicate competition means repeated sources occupy high chunk ranks before collapse; it does not mean duplicate sources survive in the final evaluation output. The next table counts distinct candidate chunks across both legs for each returned source, pooling source observations within each dataset. Earlier baseline diagnostics also retain means of query-level quantiles; these are different summaries.')
    add(table(['Dataset','Policy','Chunks/source mean','p50','p90','Union-positive lost queries','Lost positive pairs','Top10 score-tie queries'],[
        [NAMES[ds],arm,*[v['chunks_per_returned_source_pooled_within_dataset'][k] for k in ('mean','p50','p90')],v['union_positive_lost_queries'],v['union_positive_lost_sources'],v['top10_exact_score_tie_queries']]
        for ds,arms in extra.items() for arm,v in arms.items()]))
    add('BRIGHT has substantial duplication, yet removing or aggregating it has domain-dependent consequences. More distinct sources in early chunk ranks alone is not a quality gain. Attribution categories overlap and must not be added into a single failure percentage.')
    add('## 5. Experiment matrix')
    add('The [preregistered architecture and protocol](phase-c-plan.md) and [protocol hash](../evaluation/phase-c/v1/protocol.json) were frozen before C1/C2 execution. All variants use identical legs, k=60, depth 500 per leg, top 100 unique sources and no reranker. No weights, k search, query rewriting, routers or additional algorithms were tried.')
    add(table(['Policy','Per-leg source construction','Between-leg fusion / representative'],[
        ['C0','Existing chunk RRF, then first-occurrence source collapse','Existing first fused chunk'],
        ['C1','Best distinct chunk per source; compact source ranks','RRF over source ranks; R0'],
        ['C2','First two distinct chunks/source; sum 1/(60+raw chunk rank); order sources and compact ranks','Same source-rank RRF; R0'],
        ['C3','Conditional contribution cap only if redundancy gate passed','Skipped: no gate passed']]))
    add('C2 is a two-stage rank aggregation. It does not sum raw chunk scores directly between legs; using one chunk reduces it to C1. R0 distributes each final per-leg source vote among contributing chunks proportionally to their within-leg rank weights, sums chunk attribution across legs, then selects the largest contribution. New-policy ties follow first encounter in sorted-leg/input order. Existing C0 ties remain unchanged.')
    keys=('ndcg@10','recall@10','recall@20','recall@100','mrr@10')
    for ds,v in dev.items():
        add('### '+NAMES[ds])
        add(table(['Arm',*keys,*(['alpha_ndcg@10','aspect_recall@10'] if ds.startswith('bright') else [])],[
            [arm,*[x['metrics'][k] for k in keys],*([x['metrics']['alpha_ndcg@10'],x['metrics']['aspect_recall@10']] if ds.startswith('bright') else [])]
            for arm,x in v['arms'].items()]))
        add(table(['Policy − C0','Metric','Mean Δ','95% paired CI','W/T/L'],[
            [arm,m,s['mean_delta'],f"[{s['ci95'][0]:.6f}, {s['ci95'][1]:.6f}]",f"{s['wins']}/{s['ties']}/{s['losses']}"]
            for arm,stats in v['paired_vs_C0'].items() for m,s in stats.items()]))
    add('Paired bootstrap uses seed 20260911 and 10,000 query resamples separately per dataset. Intervals are exploratory; no cross-dataset pooled bootstrap or combined quality score is used. Exact per-query equality defines ties. Independent pytrec_eval checks differ by at most 2.23e-16 for development standard metrics. BRIGHT aspect metrics retain the accepted P4 conventions.')
    add('### Fusion cost and movement')
    add(table(['Dataset','Policy','Mean ms','p50 ms','p95 ms','Mean candidate chunks','Mean signed common-source rank change'],[
        [NAMES[ds],arm,*[v['fusion_ms'][k] for k in ('mean','p50','p95')],v['mean_candidate_chunks'],v['mean_common_source_rank_movement']]
        for ds,d in dev.items() for arm,v in d['arms'].items() if arm in ('C0','C1','C2')]))
    add('Cost includes deterministic Python fusion, result construction and source collapse, excluding embedding and retrieval. Each query has one warmup and three timed repetitions with rotated policy order. Lower signed rank movement means promotion among common returned sources; it is not a positive-only quality metric. No isolated peak-memory measurement was made; all three methods process at most 1,000 leg results and retain bounded per-source contributors. C1/C2 are cheaper in this implementation, but their quality tradeoff fails the shared gate.')
    add(table(['Dataset','Policy','Recovered positives: queries/pairs','Lost C0 positives: queries/pairs','Positive promotions: queries/pairs','Positive demotions: queries/pairs'],[
        [NAMES[ds],arm,*[f"{v['diagnostic_query_counts'][k]}/{v['diagnostic_source_counts'][k]}" for k in ('recovered_positive','lost_positive','positive_promotions','positive_demotions')]]
        for ds,d in dev.items() for arm,v in d['arms'].items() if arm in ('C1','C2')]))
    add('C3 required both a C2–C1 nDCG@10 decline of at least 0.01 on BRIGHT and at least 10% of queries exhibiting the preregistered near-duplicate two-chunk distractor promotion. Neither BRIGHT domain met it: C2 improved versus C1, and measured redundant-promotion queries were zero. C3 was not executed.')
    add('## 6. Representative evidence analysis')
    add('Every checked representative retains its source, exact original content slice, coordinates and document revision. The full three-policy audit checks 154,800 source representatives. C1/C2 provenance additionally records the contributing chunks and rank-based vote attribution. No synthetic chunks, merged text or reranker-based selection were introduced.')
    add(table(['Dataset','Policy','Queries with representative/leg disagreement','Per-leg disagreement pairs','Mean changed representatives vs C0'],[
        [NAMES[ds],arm,v['representative_disagreement_queries'],v['representative_leg_disagreements'],dev[ds]['arms'][arm]['mean_representative_changes']]
        for ds,d in extra.items() for arm,v in d.items()]))
    add('Disagreement is a diagnostic, not a weak-evidence label. Source qrels establish whether a document is relevant, not whether its selected chunk contains the needed evidence. Therefore representative semantic adequacy and the “positive source, weak chunk” failure rate are **not identifiable** from these labels. R1 was not needed to choose the shared policy and was not run. Phase D needs exact evidence-span judgments to evaluate this distinction.')
    add('## 7. Selected development policy')
    add('Exactly one policy is selected: **C0**. Selection was frozen at '+decision['frozen_at']+' before broader validation. The gate required SciFact nDCG loss ≤0.01, each BRIGHT loss ≤0.005, each Recall@10/20/100 loss ≤0.01, each BRIGHT aspect loss ≤0.01, and at least one BRIGHT nDCG gain ≥0.01 with a positive lower confidence bound. Prefer C1, then C2, then gated C3; otherwise retain C0.')
    add('C1 fails BRIGHT preservation and convincing-gain checks: Stack Overflow nDCG decreases by 0.010176 and its aspect metrics also cross their loss boundary. C2 improves Robotics nDCG by 0.046535, but SciFact decreases by 0.109215 and Recall@10 by 0.034444. Those regressions rule out one shared replacement. Validation does not reopen policy tuning. See [decision and individual gate results](../evaluation/phase-c/v1/development-decision.json).')
    add('## 8. Broader validation')
    add('Full query sets are fixed: NFCorpus 323, FiQA 648, BrowseComp-Plus 830. Full original corpora contain 3,633, 57,638 and 100,195 documents. Dataset revisions and individual download hashes are recorded in [acquisition revisions](../evaluation/phase-c/v1/acquisition-revisions.json) and normalized manifests. No query subsampling occurred. The generic BEIR adapter preserves original document/query IDs and official test labels. Its SciFact output remains byte-for-byte equivalent at the normalized data level to accepted P4.')
    add('FiQA contains 38 records with blank title and body, including positive document 117276. Following the user-approved exception, all raw records, queries and qrels remain intact while 57,600 nonempty records are indexed. Empty positives remain in metric denominators. This is explicitly not a claim that all 57,638 records produce vectors. The exception is independent of performance and leaves production parsing/embedding unchanged. See [exception manifest](../evaluation/phase-c/v1/empty-document-exception.json).')
    add('BrowseComp-Plus uses the pinned official corpus and all queries. Its evaluation-only adapter decodes the official query and support document IDs, verifies both evidence and gold ID sets against official GitHub qrels, and keeps labels separate from materialized documents. It does not read answers into ranking, filter by positives or invoke Agent. Original document text, IDs and URLs are preserved. Evidence-document and answer-bearing gold-document results remain separate. The [official benchmark](https://github.com/texttron/BrowseComp-Plus) and [pinned revision](../evaluation/phase-c/v1/browsecomp-official-revision.json) define these labels.')
    add('Selected C0 is identical to the current C0 control by construction, so one capture is reused instead of repeating retrieval under a new name. Selected-minus-C0 deltas are all zero, CI [0,0], and all queries tie. This verifies the retained baseline on new datasets; it does not provide an independent treatment estimate or a generalization result for rejected C1/C2.')
    for ds in ('nfcorpus','fiqa','browsecomp-plus'):
        add('### '+NAMES[ds])
        path=V/f'validation/{ds}-summary.json'
        if not path.exists():
            add('**Pending full-corpus index/capture completion. No validation score is reported.**');continue
        v=read(f'validation/{ds}-summary.json')
        experiment=read(f'validation/{ds}-experiment.json');build=experiment['index_report']
        add(f"The READY index contains {build['manifest']['document_count']:,} documents and {build['manifest']['chunk_count']:,} chunks. The normal builder reused {build['cached_inputs']:,} embedding inputs and embedded {build['embedded_inputs']:,} additional inputs; build time was {build['build_seconds']:.3f} seconds, excluding the separately recorded embedding-cache preparation. Full before/after snapshot verification passed.")
        keys2=(*(('recall@5',) if ds=='browsecomp-plus' else ()),*keys,*(('recall@1000',) if ds=='browsecomp-plus' else ()))
        add(table(['Arm','Label set',*keys2],[
            [arm,label,*[x['metrics'][k] for k in keys2]] for arm,rec in v['arms'].items() for label,x in rec['labels'].items()]))
        add(table(['Arm','Returned sources mean','min','max'],[[arm,*[rec['returned_sources'][k] for k in ('mean','min','max')]] for arm,rec in v['arms'].items()]))
        add(f'Full per-query rankings, raw legs, metric/reference checks and snapshot/environment identities are preserved in the run directory; [summary](../evaluation/phase-c/v1/validation/{ds}-summary.json), [protocol](../evaluation/phase-c/v1/validation/{ds}-protocol.json), [execution metadata](../evaluation/phase-c/v1/validation/{ds}-experiment.json).')
    add('BrowseComp official Recall@5/@100/@1000 and nDCG@10 are calculated on the available source ranking from the fixed 500-chunk legs. A semantic or BM25 leg returns at most 500 sources; fused depth is at most 1,000 and often smaller. Recall@1000 therefore measures this configuration’s actual candidate coverage, not a separately retrieved 1,000-source pool. Primary source cutoff remains 100. TREC exports use strictly decreasing rank scores to preserve tied-source ordering in independent evaluation.')
    add('Storage preparation required a user-authorized external APFS sparse image. Qdrant host-bind mounts reported incompatible FUSE storage and failed before retrieval. The successful runs use a separate native Docker volume with the same Qdrant 1.19.0 image and index parameters. Existing services and historical READY snapshots were not changed. Failed attempts are retained and produced no scored rankings. Cache staging uses the exact production chunker, tokenization, embedding model and SQLite keys; the normal builder validates and publishes the final snapshot. See [storage incident](phase-c-storage-note.md).')
    add('## 9. Final product decision')
    add('**Decision A — keep current chunk-level Hybrid fusion.** '+
        ('The frozen development decision has completed broader validation without further tuning. ' if finished else
         'This is the frozen development decision; Phase C completion still requires every pending validation/verification/archive item reported here. ')+
        'The alternatives offer a CPU cost improvement and some BRIGHT gains but fail the shared quality requirements. No dataset-specific routing, BM25 disabling, new RRF parameter or forced source-level redesign is justified. Production remains unchanged; experimental policies are confined to evaluation code.')
    add('## 10. Tests')
    if tests:
        add(table(['Suite/subset','Cases','Passed','Failed','Errors','Skipped'],[
            [name,*[v[k] for k in ('cases','passed','failed','errors','skipped')]] for name,v in tests.items() if isinstance(v,dict) and 'cases' in v]))
        add(f"Freshness/update checks: {tests['freshness']['passed']} passed, {tests['freshness']['failed']} failed.")
    else:add('Final test evidence is being assembled. Individual completed runs are retained in `evaluation/results/phase-c-v1`; do not infer suite counts from this pending table.')
    add('Subsets overlap: evaluation, Phase A, Phase B and new Phase C cases are contained in the deterministic/service suites and must not be summed. Phase A covers Agent contracts, exact match/read, runtime and CLI. Phase B covers reranker input budgets, stable scoring/fallback and real long-query inference. New Phase C tests exercise source aggregation, rank semantics, identity/provenance, filtering, ties, source uniqueness, representatives, BEIR/BCP normalization and the explicit empty-document denominator rule. Freshness uses real SQLite/Qdrant/model services to verify old indexed snapshots alongside changed live reads, stale references, edits, renames, deletes and publication.')
    add('The source scope audit reports zero non-evaluation production changes against `75ba733`. Source-fusion ranking/statistics and validation ranks/metrics are also replayed from archived legs with zero new model calls. Test seams follow the requested public source-fusion and external-adapter boundaries. Red/green traces are retained.')
    add('## 11. Remaining issues')
    add(table(['Area','Evidence and next question'],[
        ['Candidate generation','BRIGHT Robotics union recall is 0.8540 at 500 chunks/leg; missing evidence cannot be recovered by fusion. Lexical-only positives remain useful.'],
        ['Fusion','The union-to-top100 gap is real, but C1/C2 do not provide a robust shared replacement. Source duplication alone is an insufficient optimization target.'],
        ['Chunk representation','Exact provenance is verified, but source relevance cannot establish representative-chunk adequacy. Need evidence-span labels.'],
        ['Reranker','B3 compatibility/regression is preserved; no reranker quality claim is inferred from rerank-off Phase C experiments.'],
        ['Agent policy','No BrowseComp Agent evaluation, query rewriting or policy learning ran. Retrieval quality does not establish end-to-end task success.'],
        ['Evaluation','Public labels are exposed, confidence intervals exploratory, qrels incomplete for unjudged documents, and no ARKB-native production distribution has been measured. FiQA has the explicit 38-empty-record exception; BCP cutoff metrics reflect actual available depth. Full BCP indexing is an operational scale/cost issue and must not be hidden by corpus reduction.']]))
    add('## 12. Phase D handoff')
    add('Stable inputs for Phase D are the retained chunk Hybrid baseline, unchanged Phase A tool/execution contract, unchanged Phase B B3 reranker, exact source/revision/span provenance, reusable official-data adapters, and frozen candidate-loss diagnostics. Native dataset construction should distinguish a missing candidate, a rank loss, a relevant source represented by the wrong span, and stale/live evidence behavior. Separate development and held-out native queries; preserve annotator judgments, exact evidence spans and versioned snapshot identities. These are handoff requirements, not a Phase D implementation.')
    add('Phase E must reuse `/Volumes/ARKBPhaseC/data/browsecomp-plus` and its completed Phase C SQLite/Qdrant snapshot, original 830 query IDs and frozen retrieval configuration. Do not regenerate queries or redefine corpus scope. The external sparse image lives at `/Volumes/闪迪1T/arkb-phase-c-v1/arkb-phase-c.sparseimage`; mount it at the recorded location for live document access. Archive manifests record exact hashes and restore paths. Full BrowseComp query text and raw retrieval evidence stay in local controlled artifacts rather than public report tables.')
    add('Reproduction: use a fresh output path with `evaluation/experiments/validate_phase_c.py`; model/service access is required only for original indexing/capture. `evaluation/audits/replay_phase_c.py` and `replay_phase_c_validation.py` reproduce saved ranks, provenance and statistics offline. Source/configuration copies and hashes accompany each run. Existing P4/Phase A/Phase B evidence is preserved.')
    (ROOT/'docs/phase-c-report.md').write_text('\n'.join(parts))

if __name__=='__main__':main()

# ARKB: current system architecture and evaluation evidence

Review scope: checkout `c140b2e`, including the pre-existing, untracked Phase C validation diagnostics and agentic tool-selection plan. Current production code takes precedence over historical reports. Historical measurements below retain their original configurations and limitations. This review did not change production code, indexes, models, or experiment inputs.

## 1. ARKB in One Paragraph

ARKB is a local knowledge-retrieval system in which an LLM controls an investigation over a Markdown knowledge base. It can locate exact occurrences, request lexical or semantic rankings, combine those rankings, open evidence, reformulate a query, and decide whether to answer or acknowledge missing information. The retrieval infrastructure owns reproducible execution and provenance; the agent owns the choice and sequence of actions. SQLite preserves versioned text snapshots and reusable embeddings, while Qdrant performs vector retrieval. Current defaults are Qwen3-Embedding-0.6B, a Qwen3.5-4B agent with thinking enabled, semantic search when no mode is specified, and optional reranking disabled for normal `ask`. The strongest evidence supports specific retrieval and reliability improvements. It does **not** yet establish that the current agent produces better grounded answers than a comparable fixed RAG pipeline.

## 2. System Architecture

```text
                           KNOWLEDGE PREPARATION
Flat Markdown directory
  -> load title + normalized body
  -> section-aware, token-budgeted chunks + source/revision/span identities
  -> embedding input preparation -> reuse cache / embed missing inputs
  -> validate candidate SQLite snapshot and Qdrant collection
  -> atomically publish the active snapshot pointer

                              QUERY EXECUTION
CLI / Python API -> Runtime
                     |
                     +-> Direct deterministic retrieval -> ranked evidence
                     |
                     +-> Agent: choose tools, queries, modes, limits, stopping
                            |
                 +----------+----------------------+------------------+
                 |                                 |                  |
               match                             search              read
                 |                                 |                  |
           live Markdown                  captured index snapshot  live Markdown
           exact occurrence                BM25 / semantic / RRF   expand evidence
                 |                                 |                  |
                 +------------ validated evidence references --------+
                                           |
                                      Agent observes
                                           |
                            search/read again or finish/finalize
                                           |
                          answer + status + citations + diagnostics

Separate Python generation workflow:
ranked evidence -> context packing/token budget -> one generation -> citation checks
```

The architecture has six important responsibilities, rather than one mandatory pipeline:

| Responsibility | What it owns | What it does not decide |
| --- | --- | --- |
| Knowledge access and indexing | Parsing, chunk boundaries, identity, revisions, embedding compatibility, cache reuse, publication | Whether a query needs more evidence |
| Deterministic retrieval | Exact matching, BM25, vector search, rank fusion, optional reranking | Query reformulation or answering |
| Runtime composition | Resource ownership, lazy service initialization, binding tools to a knowledge scope and snapshot | An automatic best-mode router |
| Agent control | Tool selection, query text, supported search mode, result limit, expansion, answer/abstention | The ranking algorithm inside each tool |
| Tool session and observation | Argument validation, evidence references, delivery accounting, recovery and termination mechanics | Whether a citation actually entails a claim |
| Explicit generation and evaluation | Context construction, citation structure, measurements and comparisons | A replacement agent policy |

This separation is real in the implementation. `Runtime.ask` captures a manifest and composes lazy BM25/semantic capabilities; `RetrievalEngine.search` dispatches an explicitly supplied mode. Hybrid invokes both legs with the same query and source filter, verifies matching snapshot identities, and fuses their rankings. It neither rewrites the query nor silently changes modes when a leg fails. The two Hybrid legs currently execute sequentially. The agent can ask for several tools in one model response, but their execution is also sequential. [Runtime](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/runtime.py), [retrieval engine](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/retrieval/engine.py), [Hybrid implementation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/retrieval/hybrid.py).

“Deterministic” describes the retrieval control flow and explicit configuration. Embedding and reranker inference still depend on model artifacts and numerical environments, and approximate vector search is not guaranteed to reproduce identical results. Qdrant exact-versus-ANN selection is independent of literal `match`.

### The three evidence tools

| Tool | Appropriate need | Data and behavior |
| --- | --- | --- |
| `match` | “Which notes literally mention this term?” | Live normalized body or filename matching. Body results are occurrences, not distinct documents. Ordinary case-sensitive literals use an in-process scan; regex/case-insensitive matching uses one batched ripgrep invocation. |
| `search` | “Find information relevant to this concept/question.” | Ranked chunks from the captured snapshot, using BM25, semantic, or Hybrid. The agent sees relevance order, source, title, content, and opaque references; low-level scores and provenance remain in underlying results/diagnostics. |
| `read` | “Open the evidence or inspect its surrounding document.” | `read(ref=...)` expands the whole live document by default; `expand="snippet"` rereads the bound excerpt. `read(source=...)` opens a known filename. |

Agent-facing `read` does not expose arbitrary offsets or separate document/section IDs. Low-level Python document access still supports those validated selectors. There is no automatic neighboring-chunk expansion or graph traversal: further reading is chosen by the agent. [Tool contract](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/agent/tools.py), [reference session](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/agent/session.py), [exact retrieval](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/retrieval/exact.py).

### Retrieval choices and defaults

BM25 uses in-memory postings built from the SQLite snapshot’s title and body, with NFC/casefold word tokenization, unique query terms, `k1=1.2`, and `b=0.75`. It needs no model or vector service at query time. There is no stemming, learned sparse retrieval, or multilingual segmentation.

Semantic retrieval embeds the supplied query, with the saved retrieval instruction, into the snapshot’s compatible embedding space and obtains cosine-ranked IDs from Qdrant. SQLite hydrates those IDs into exact text and source locations. The supported persistent embedding path is specifically the validated `qwen3-embedding:0.6b` pairing, not an arbitrary model selected by a tag flag.

Hybrid uses chunk-level reciprocal rank fusion, `sum(1/(60 + rank))`, avoiding direct addition of incomparable BM25 and cosine scores. Each chunk gets at most one vote per leg. Normal Hybrid candidate depth is 20 per leg; agent search defaults to five returned results. **Production does not collapse chunks into unique sources.** Optional Qwen3-Reranker-0.6B scores a bounded pool after retrieval; the host composition controls whether it is enabled, not an agent tool argument. [Configuration](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/config.py), [BM25](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/retrieval/bm25.py), [semantic retrieval](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/retrieval/semantic.py), [reranker](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/retrieval/qwen_rerank.py).

### Interfaces and present scope

The five product CLI commands are `index`, `status`, `match`, `search`, and `ask`. They delegate to Runtime; JSON and trace options change presentation, not retrieval policy. Python additionally exposes prepared engines, tools, observers/budgets, historical snapshots, and explicit generation. **There is no MCP server in the current source or dependency/entry-point configuration. MCP is planned.** It cannot be described as an implemented third interface.

The maintained product scope is English retrieval over UTF-8 `.md` files directly inside one directory. Subdirectory ingestion, general PDF/HTML ingestion, automatic synchronization, persistent conversational memory, and a knowledge graph are not part of the current path. Historical multilingual datasets remain as historical evidence. [CLI](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/interfaces/cli.py), [package configuration](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/pyproject.toml), [English scope](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/english-scope-cleanup.md).

## 3. End-to-End Query Flow

### 1. Ingestion and indexing

**Start:** `index` or the Python builder receives the complete flat knowledge scope.

The knowledge layer scans files, checks that they did not change during scanning, separates the first H1 title from the stripped body, and produces verbatim body slices. The current recursive algorithm is `markdown-v2`: heading-aware sections, block packing, recursive splitting of oversized blocks, a 512-token body budget, and up to 64 tokens of overlap within a section. Coordinates refer to this loaded body, not raw-file byte offsets.

The embedding layer assembles title plus body and checks the complete input against its embedding limit, normally 8,192 tokens, without silent truncation. Source filenames and offsets are metadata, not embedding text. Identical prepared texts under a compatible embedding specification reuse cached vectors, including across source renames when the actual embedding input stays unchanged.

The builder creates a candidate snapshot and separate Qdrant collection, checkpoints successful embedding batches, stores chunk records, uploads vectors, and verifies content/vector/payload consistency and readiness. Only then does a SQLite transaction change the active pointer. Failure preserves the previous READY snapshot and successful cache entries; historical READY collections remain available to captured readers.

**Decisions:** all deterministic; no generative LLM decides chunk boundaries or publication. An embedding model supplies vectors.

**Termination:** a verified READY snapshot, a verified unchanged-index reuse, or an explicit failed build. Incremental behavior primarily reuses embeddings; a changed corpus still publishes a new complete snapshot rather than editing the active collection in place. [Index builder](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/knowledge/indexing.py), [chunker](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/knowledge/chunking.py), [storage](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/knowledge/sqlite.py).

### 2. Deterministic retrieval

**Start:** a direct `match` or `search`, or an equivalent Python request.

Exact matching reads live documents and returns located occurrences. Ranked search validates parameters, captures a published snapshot, executes the chosen ranking strategy, optionally reranks, and returns chunks with scores and provenance. Source filters apply before top-K. A standalone BM25 request can run with SQLite alone; indexing itself currently includes embeddings/Qdrant.

**State passed:** query, explicit strategy/settings/filter, snapshot identity, and SearchResults containing source, revision, span, text, and score semantics.

**Termination:** the result list or an explicit error. There is no query reformulation or generation in this workflow.

### 3. Agentic retrieval and evidence expansion

**Start:** `ask` creates a fresh conversation and tool session, retaining one search snapshot throughout that invocation. Services are loaded only if selected capabilities need them. Direct replies and live tools can work without a published index; ranked search cannot.

The model receives the question and available tool schemas. It chooses a tool, query, mode, source restriction, and limit. The session validates arguments and converts results into run-scoped evidence references. Those references retain hidden source/revision/span/snapshot bindings and can only be cited once delivered.

Tool observations return to the conversation. The model can use a discovered term in another search, change retrieval mode, read a document, or conclude. That is where agentic behavior resides: **the next query can depend on the evidence returned by the previous query.** There is no separate automatic query-rewriting stage inside retrieval.

Reading a reference resolves a single live document, checks its identity and revision, and verifies the original excerpt. Edits, deletion, or renaming produce recoverable errors. Reading a known filename can refresh live evidence. Repeating indexed search alone does not refresh a stale snapshot; reindexing is required for ranked search to see changed content.

An exact repeated-evidence detector adds a stopping reminder after sustained unproductive searches, allowing one unproductive follow-up. It recognizes changed chunks, ranges, revisions, or content as progress. It is a prompt intervention, not a semantic novelty model or forced stop. [Agent loop](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/agent/loop.py), [session](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/agent/session.py).

### 4. Answering and termination

During normal collection, the model can call `finish(answer, status, evidence_refs)`. The last of the default eight model requests is reserved for schema-constrained finalization with evidence tools disabled. Collection-budget exhaustion also transitions to finalization using previously delivered evidence; it does not grant extra model turns.

The optional observer can limit tool attempts, query calls, reads, cumulative evidence tokens, and elapsed time. It charges duplicate evidence again and withholds an entire over-budget observation rather than silently trimming it. These extra budgets are Python composition options; ordinary CLI `ask` exposes the turn limit, not this full budget policy. Reference-token accounting is not an exact bound on the generation model’s rendered context window.

Expected tool misuse yields recoverable observations. Unexpected backend failures terminate with an error, preserving partial state. Deadlines are cooperative for ordinary synchronous provider work; the exact matcher has its own bounded/cancellable implementation. The evaluator’s additional hard timeout is not a universal product deadline.

**Termination:** canonical answer status (`answered`, `partial`, or `insufficient_evidence`) with resolved citations, or an explicit runtime error and partial diagnostics. Citation validation checks identity and current revision, not semantic support. A no-tool response to a knowledge-dependent question is still possible: retrieval need is a model decision, and the no-reference rejection applies after retrieval has been attempted. [Observation/budgets](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/agent/observation.py), [final validation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/agent/session.py).

### 5. Separate fixed generation

Python callers can take retrieved results, validate their snapshot provenance, deduplicate/merge overlapping evidence, pack whole candidates into a measured context budget, generate once, and validate structured claim citations and optional verbatim quotes. This path has a pinned generation-token counting profile and uses `think=False`; it is not automatically called by `ask`. Even valid quotes do not establish entailment. [Context construction](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/generation/context.py), [generation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/generation/generate.py), [citation validation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/generation/citations.py).

### 6. Evaluation

**Start:** a versioned dataset and frozen experiment protocol. Runners validate corpus/model/configuration identities, prepare or restore isolated snapshots, and call the same production capabilities with ordinary queries. Gold labels remain on the scoring side.

**Stages:** capture raw rankings or agent trajectories → reconstruct evidence actually returned/delivered/submitted → score the appropriate outcome → retain failures and denominators → compute paired comparisons → verify checksums and replay → report validity and adoption status separately.

**Termination:** a complete experiment artifact or a preserved interrupted/failed attempt. A completed script is not synonymous with a successful agent, improved quality, or release eligibility.

## 4. Major Design Decisions

“Strong” below means convincing evidence for the stated, bounded conclusion—not proof across all production workloads.

| Decision | Motivation and mechanism | Evidence and judgment |
| --- | --- | --- |
| Agent control over independent retrieval tools | Different requests need literal lookup, conceptual discovery, document expansion, or several dependent searches. Put sequencing in the model while retaining directly callable retrieval. | **Partial.** Historical coverage improves on some exploratory/discovery cases, but tools, query wording, number of calls, reads, and generation all vary together. No completed controlled current-agent versus fixed-RAG answer-quality result. |
| Separate `match`, `search`, and `read` | Literal occurrence, relevance, and expansion have different semantics and costs; one ranking API cannot represent all three honestly. | **Strong contract evidence; partial policy evidence.** Tests verify semantics. The historical RAG-enumeration failure shows five occurrences can cover only two of five required documents; it does not prove the agent uses these distinctions reliably. |
| Immutable snapshots plus live document access | Keep lexical/vector evidence consistent through a session while allowing current-file lookup and explicit stale-evidence detection. | **Strong correctness evidence.** Publication, edit/delete/rename, captured-reader, and corruption tests; 16 current Phase A/C live freshness checks reported. No measured automatic freshness SLA or answer-quality ablation. |
| Section-aware chunking and exact provenance | Fit embedding inputs while preserving source slices and stable evidence identities; enable reliable reading, caching, and evaluation. | **Strong invariants; no meaningful quality optimum established.** Tests cover lossless reconstruction, budgets, overlap, and identity reuse. No controlled evidence establishes 512/64 or this chunker as best for answers. |
| BM25 + semantic + chunk RRF | Preserve complementary lexical and conceptual candidates without calibrating raw score scales. | **Strong but domain-specific retrieval evidence.** Gains on SciFact and technical domains; clear FiQA loss. No experiment establishes RRF `k=60` as optimal. |
| Keep production chunk fusion after source-fusion experiments | Source aggregation might reduce competition from repeated chunks. C1 keeps one chunk/source per leg; C2 combines two before source-rank fusion. | **Strong evidence against adopting these shared replacements.** C2 helps Robotics but badly regresses SciFact; C1 fails preservation/gain gates. Neither is connected to production. |
| Stable reranker ties and protected body allocation | Old long queries consumed the 512-token input, eliminating document text and creating identical scores; identity tie-breaking then scrambled useful Hybrid order. | **Strong tie-policy evidence; partial incremental allocation evidence.** Fixed-score replay isolates tie handling. Body preservation eliminates zero-body inputs, but its extra ranking gain over stable ties is uncertain. |
| Opaque references, recovery, reserved finalization | Models confused IDs/ranges; one invalid call aborted runs; collection could consume the opportunity to answer. | **Strong contract evidence; small-sample reliability evidence.** Phase A final outputs improve from 5/8 to 7/8. Changes were a bundle; not proof of improved factual accuracy. |
| Repeated-search reminder and thinking default | Discourage unproductive searching while enabling evidence-dependent follow-up. | **Partial/local.** A reproduced loop finishes 3/3 times with the reminder; a larger P3 toggle comparison shows no aggregate benefit. Thinking recovers facts in a tiny cross-document control but is not broadly isolated. |
| Default 4B with configurable agent model | Offer local agent execution while permitting capacity/cost tradeoffs. | **Historical model effect measured.** 27B improves v1 task completion, but increases latency and does not uniformly improve recall or calls. Current contract was not used in that full matrix. |
| Durable embedding cache and bounded upsert allocation | Avoid repeated model work and recover expensive indexing without publishing partial indexes. | **Strong operational/invariant evidence.** Durable recovery and identical-vector reuse are verified; the upsert refactor reduces fixture peak temporary allocation 41.7%. No relevance gain is implied. |

Evidence sources: [Phase A](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-a-report.md), [Phase B](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-b-report.md), [Phase C](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-c-report.md), [P3](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/p3-controlled-results.md), [model ablation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase1-agent-model-ablation-results.md), [indexing tests](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/tests/knowledge/test_indexing.py).

## 5. Evidence-Backed Optimizations

### Hybrid has demonstrated value, but not universal superiority

The following source-level nDCG@10 measurements use fixed 500-chunk legs and source projections. They are **not** measurements of the default five-result agent request. The reranked column uses the repaired Phase B B3 allocation, whose B5 fallback evaluation has the same rankings on these datasets.

| Dataset | Queries | BM25 | Semantic | Current Hybrid/C0 | Hybrid + repaired reranker |
| --- | ---: | ---: | ---: | ---: | ---: |
| SciFact | 300 | 0.6608 | 0.6961 | 0.7216 | 0.7666 |
| Bright-Pro Stack Overflow | 115 | 0.3360 | 0.3434 | 0.4807 | 0.4075 |
| Bright-Pro Robotics | 101 | 0.2480 | 0.3140 | 0.3904 | 0.3558 |
| NFCorpus | 323 | 0.3061 | 0.3540 | 0.3627 | Not evaluated here |
| FiQA | 648 | 0.2337 | 0.4449 | 0.3697 | Not evaluated here |

Hybrid’s benefit has a concrete candidate explanation: BM25 contributes 34 positive query-document pairs absent from the captured semantic leg on Stack Overflow, and 66 on Robotics. Semantic adds 55 and 88 absent from BM25. Both legs contribute useful evidence.

FiQA shows the opposite aggregate ranking outcome: Hybrid minus Semantic nDCG@10 is **−0.075255**, exploratory paired 95% interval **[−0.095104, −0.055618]**, with 146 query wins, 241 ties, and 261 losses. NFCorpus’s +0.008733 interval includes zero. These are post-hoc validation diagnostics; they justify investigating ranking preservation, not installing a dataset-specific router after looking at validation labels. [Development summaries](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-c/v1/development-summary.json), [FiQA](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-c/v1/validation/fiqa-summary.json), [NFCorpus](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-c/v1/validation/nfcorpus-summary.json), [supplementary diagnostics](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-c-validation-diagnostics.md).

### Reranker repair: distinguish the repair from enabling reranking

Old inputs omitted every candidate body for 30/115 Stack Overflow queries and 26/101 Robotics queries. With the **same saved model scores**, changing only tie handling to preserve incoming order raises nDCG@10:

- Stack Overflow: **0.341227 → 0.398871**.
- Robotics: **0.291856 → 0.348986**.
- SciFact: unchanged.

This is one of the cleanest causal comparisons in the repository. B3 then caps the query at 128 head-plus-tail tokens, title at 64, and guarantees at least `min(body length, 128)` body tokens within the 512-token frame. Zero-body inputs disappear. However, B3’s extra nDCG gain over the stable-tie control is only +0.008658 and +0.006825, with both intervals crossing zero. The input invariant improved more convincingly than the incremental ranking metric.

Relative to **plain Hybrid**, repaired reranking still loses 0.073123 nDCG on Stack Overflow and 0.034603 on Robotics; both reported intervals are below zero. SciFact gains 0.044947, interval [0.021844, 0.068412]. Reranking costs about 4.5–4.6 seconds/query for 20 candidates on the measured CPU setup. Keep it optional. These frozen experiments rerank the first 20 unique sources and append positions 21–100 unchanged; unchanged Recall@100 is therefore by construction, not proof that reranking protects evidence. Production reranking instead operates on chunks. [Phase B report](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-b-report.md), [machine summary](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-b/v1/summary.json).

### Candidate growth and source aggregation are not sufficient by themselves

P3 increased per-leg depth 20 → 40 while holding the fused pool at 20 and returned chunks at 10. Raw candidate evidence coverage rose **95.35% → 98.84%**, but final coverage stayed **84.88%**. Needed facets appeared at fusion ranks 22 and 33, outside the retained pool. These provisional labels support a stage-local bottleneck diagnosis, not a universal rejection of deeper retrieval.

Phase C independently tested source aggregation on fixed legs. C2 raises Robotics nDCG **0.390414 → 0.436949**, but drops SciFact **0.721640 → 0.612424**. C1 also fails the shared selection gate. C0 is retained. This says the tested replacements are unsuitable as one shared policy; it does not establish that C0 is optimal. [P3 results](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/p3-controlled-results.md), [Phase C selection](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-c/v1/development-decision.json).

### Agent reliability improved; answer superiority remains unproven

Phase A compares eight registered cases under the old/new execution contracts:

| Outcome | Old | New |
| --- | ---: | ---: |
| Runs with final answers | 5/8 | 7/8 |
| Strict MuSiQue structured predictions | 1/4 | 4/4 |
| Fatal collection-tool errors | 1/34 | 0/34 |
| Serializable canonical envelopes | Unavailable | 8/8, including one error |

The new contract handles a bad reference recoverably and successfully finalizes some budget/turn-bound runs. It still has one invalid finalization output. Raw MuSiQue answer F1 **falls from 0.5000 to 0.0786**, while support F1 rises to 1.0. The new answers contain explanations rather than only short benchmark answers, creating a declared format confound. Neither serialization nor support retrieval proves factual correctness.

A separate exact-call replay on the same 109,188-document corpus finishes in **19.89 s with zero subprocesses**; the old call was interrupted after **173.94 s**. This supports eliminating per-document subprocess overhead. The new eight-case agent arm made no `match` calls, so its completion improvement cannot itself measure matcher acceleration. [Phase A summary](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-a/reliability-v1/summary.json), [report and replay context](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-a-report.md).

The historical 360-run model comparison reports:

| Model | v1 task success | Exploratory success | Source recall | Mean task time |
| --- | ---: | ---: | ---: | ---: |
| 4B | 77.5% | 40% | 85.88% | 14.25 s |
| 9B | 85.0% | 80% | 89.61% | 21.84 s |
| 27B | 95.0% | 100% | 88.53% | 66.86 s |

Larger models improved historical completion/stopping, but recall and tool counts did not improve monotonically. This matrix predates the current contract and uses the mostly Chinese-query v1 dataset. It supports model capacity as a possible bottleneck, not changing the present default or claiming current English answer accuracy. Query reformulation, mode selection, and reading were not individually ablated. [Model results](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase1-agent-model-ablation-results.md), [raw comparison](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/results/agent_model_ablation/20260909-phase1-formal/comparison.json).

The repeated-search reminder fixed a particular failure in three repeated runs, but P3’s controlled toggle yields **14/24 finals and 5.0 tools/run in both arms**; it fires in only two enabled runs. Thus there is local regression evidence and no demonstrated broad efficiency gain. Those runs also predate reserved finalization. [Stopping validation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/agent-search-stopping-validation.md), [P3 toggle experiment](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/p3-controlled-results.md).

### Operational improvements have a separate evidence standard

The current Qdrant writer constructs one batch’s request objects at a time. A 4,097-record, 1,024-dimensional fixture preserves the identical serialized request stream and full prevalidation while reducing traced temporary peak allocation **172,957,774 → 100,787,212 bytes (41.7%)**. Whole-matrix validation remains; this is not a full-corpus peak-RAM measurement. [Allocation audit](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-c/v1/upsert-after.json).

Separate experimental probes found 2.78× faster CPU input preparation on 26 documents, 16.8% higher embedding throughput with two Ollama instances on 256 inputs with exact saved-vector agreement, and MLX float16 **25.02 versus 16.47 inputs/s**. The first is an operational preparation path, the two-server mode is not enabled in the full job, and MLX changes backend/precision and vectors without a retrieval-quality comparison. None establishes an end-to-end production speedup or justifies mixing new vectors into the existing cache. [Execution note](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-c-execution-note.md), [GPU investigation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/embedding-throughput-investigation.md).

## 6. Evaluation Framework and What It Proves

### Datasets answer different questions

| Track | Scope | Intended architectural question | Present limitation |
| --- | --- | --- | --- |
| v1 | 40 tasks over 40 example notes; exact, semantic, direct-read, exploration, QA, no-retrieval | Can the agent collect labeled sources, use required reads, respect constraints, and finish? | Mostly Chinese queries; historical/small; success does not grade answer text. |
| v2 provisional pilot | 60 queries, 31 intent families, 58 documents | Are actual evidence spans and required multi-hop combinations delivered? What changes with budgets and candidate selection? | Labels remain provisional, independent review absent, related variants are not independent tasks. |
| SciFact | 300 queries, 5,183 documents | Compare lexical, semantic, fusion, and reranking for scientific evidence retrieval | Public test has become exposed development data; not an ARKB release holdout. |
| Bright-Pro technical domains | Stack Overflow: 115/109,188; Robotics: 101/63,920 queries/documents | Long technical questions, ranking, complementary candidates, aspect coverage | Uses Bright-Pro annotations, not an interchangeable classic-BRIGHT reproduction; source labels cannot prove passage sufficiency. |
| MuSiQue diagnostic | 100 answerable/unanswerable pairs, 200 variants; original 20 paragraphs per variant | Can the agent collect multiple hops, answer, and recognize missing support? | Selected 2/3/4-hop diagnostic distribution, not full official dev; historical run has major execution/format failures. |
| Phase C validation | NFCorpus: 323/3,633; FiQA: 648/57,638 | Does a frozen retrieval policy remain useful outside the development domains? | C0 selected; alternatives not retuned on these labels. FiQA indexes 57,600 nonempty records; all qrels/denominators remain intact. |
| BrowseComp-Plus | Planned full 830 queries / 100,195 documents | Full-corpus difficult retrieval and scale | Repository status/artifact inventory still marks full validation and archival pending; no completed score or agent result is available here. |

T2’s 500-query Chinese retrieval experiment is retained history; its importer and tokenizer roadmap are retired under the English scope. The newer tool-selection/full-agent-versus-fixed-RAG document is a **future execution plan**, not evidence that those comparisons have completed. [Dataset/status documentation](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/README.md), [P4 protocol/results](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/p4-external-results.md), [Phase C status](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/phase-c/v1/artifact-locations.json), [pending tool-selection plan](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/agentic-tool-selection-evaluation-plan.md).

### What each measurement establishes

**Ranking metrics** answer whether a fixed request places judged sources near the top. Recall measures coverage; MRR rewards the first relevant result; nDCG measures relevance and position; technical aspect metrics check coverage/diversity of the supplied aspects. The external adapters use their official label conventions and linear gain, while v2 uses explicit 0–3 grades and its own relevance threshold/exponential-gain convention. Scores should not be pooled across these contracts.

**Evidence metrics** ask whether the model actually received the needed text. V2 checks exact revision and body ranges and supports OR alternatives of AND-combined spans for each weighted facet. This distinguishes a relevant document from a complete evidence chain. P3 illustrates why: reranking improves a provisional source nDCG score while removing bridge passages and lowering complete evidence coverage.

**Agent metrics** separate protocol completion, source coverage, required reading, unnecessary retrieval, attempts, turns, errors, and costs. New observation artifacts distinguish returned evidence, evidence admitted into conversation, and evidence submitted in a model request. These sets can differ. Gold-defined “evidence sufficient” points are offline diagnostics, not runtime stopping rules.

**Answer metrics** are a separate layer. MuSiQue has strict answer/support/answerability and paired-sufficiency measures. Native v2 output packets bind answer, rubric, evidence, and stop state with a hash, but meaningful semantic success requires supplied human judgments. Separate correctness/support judges and calibration machinery exist; saved calibration has zero cases and is not passed. Quote presence and valid reference IDs are not treated as semantic proof. [v2 scorer](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/evaluation/v2.py), [output review](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/src/arkb/evaluation/outputs.py), [calibration status](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/evaluation/reviews/p2-budget-20260910/correctness-calibration-status.json).

There is also a current compatibility caveat: the native v2 output scorer marks any recorded budget exhaustion as a constraint failure even if Phase A finalization successfully returns an answer. That is a strict budget-success definition, distinct from “answer produced.” Future comparisons must report both rather than silently treating the old metric as the new completion contract.

### Reproducibility strengths and limits

The framework preserves dataset/corpus fingerprints, model digests and tokenizer identities, measured source hashes, frozen protocols, snapshot identities, raw rankings/scores, model/tool traces, failure records, and independent offline replay. Some experiments rotate arm order, use exact Qdrant search to remove ANN variation, freeze candidate pools, and perform paired bootstrap by query, intent family, or MuSiQue pair as appropriate. Trials remain in denominators; missing values are not invented as zeros or successes. Official/reference scorer checks detect implementation mistakes. Historical omissions, such as missing pre-run SciFact weight hashes, remain disclosed rather than retroactively claimed.

These controls establish that a number corresponds to the recorded experiment. They cannot certify label completeness, reviewer independence, unseen production generalization, model pretraining contamination, or isolated production latency. Repeated trials at temperature zero are not additional independent information needs. Most intervals are exploratory and uncorrected for multiple comparisons.

During this review, **1,193 deterministic repository tests passed; 60 service tests were excluded**. The current-code full-corpus SciFact BM25 replay reproduced **300/300 rankings with zero metric difference**. Live model/vector benchmarks were not rerun. The initial unrestricted pytest collection aborted inside an archived MLX probe dependency; explicit `tests/` collection passed. This is a local collection-scope issue, separate from product quality. [Fresh replay result](/private/tmp/arkb-review-scifact-20260913-01a09978.json), [CI definition](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/.github/workflows/evaluation.yml).

### What remains unidentified

The completed experiments cannot yet isolate the answer-quality contribution of tool availability, query reformulation, repeated search, `read`, or stopping policy under the **current** agent contract and comparable budgets. They do not establish optimal chunk sizes, embedding models, RRF weights, ANN settings, or context budgets. Source-level public qrels cannot assess the chosen representative chunk. There is no independently reviewed native release core or completed calibrated grounded-answer comparison. The 500-chunk benchmark pools also do not establish quality at the product’s default candidate depth.

## 7. Remaining Weaknesses and Optimization Opportunities

### High-confidence priorities supported by observed failures

These are high-confidence **problems to address**. Their proposed solutions still require the experiments listed.

| Current limitation | Proposed direction and why it may help | Existing evidence | Validation experiment |
| --- | --- | --- | --- |
| Existing evidence is lost before the delivered cutoff | Investigate fusion/pool selection that better preserves strong candidates; locate the losing stage before increasing depth | FiQA union positive coverage 91.60%, C0 Recall@100 74.92%; NFCorpus 56.32% versus 33.53%; P3 facets at ranks 22/33 | On development data, freeze legs and candidate budgets; compare a small preregistered set of ranking/pool policies, then use fresh validation. Check ranking, exact evidence coverage, and downstream answers. |
| Candidate generation itself misses evidence | Treat candidate recall separately from sorting, especially for medical/technical queries | NFCorpus lacks 43.68% of positive evidence in the captured union on a macro basis; Robotics union recall is 85.40% | Compare depth and query reformulation separately with embeddings/fusion held fixed; record union recall, final coverage, and total query cost. Reranking alone cannot recover absent candidates. |
| Literal enumeration truncates occurrences without expressing completeness | Add explicit truncation/completion information or a distinct-source enumeration/pagination capability while preserving occurrence matching | Historical `exact_001` finds only 2/5 required sources because four of the first five occurrences belong to one note; this occurrence-limit semantic remains | Controlled known-total exact queries across repeated-occurrence distributions. Compare default agent completion, distinct-source recall, occurrence correctness, calls, and runtime. |
| Answer-quality uncertainty prevents rational optimization | Build independently reviewed English intent families with minimal spans, alternative evidence, bridge requirements, and answer rubrics | Pilot is 60 queries but only 31 unreviewed families; current calibration has zero cases; source improvements can accompany evidence losses | Blind paired output review, independent validation families, calibrated judges only after agreement checks; separate correct answers, support, refusal, and execution. |
| Large input preparation and validation are expensive | Extend proven cache/preparation reuse and profile remaining full-memory stages without changing embedding identities or publication guarantees | 2.78× preparation sample speedup; durable recovery; 41.7% upsert fixture allocation reduction, but full-matrix validation remains | Same complete corpus/input digest/vector cache; compare wall time, peak memory, recovery after interruption, exact requests, and READY snapshot equality. |

For NFCorpus, the raw union-to-top100 gap is not all recoverable by sorting: the saved oracle identifies about 0.23 percentage points of top-100 capacity loss and 22.56 points of ranking headroom. Such gold-informed bounds diagnose opportunity; they are not achievable-performance promises. [Validation diagnostics](/Users/daboluo/MyWorkSpace/GitHub/agentic-retrieval-for-knowledge-bases/docs/phase-c-validation-diagnostics.md).

### Reasonable hypotheses requiring evaluation

1. **Measure the value of the agent’s action space.** Compare the same model with full tools, restricted retrieval modes, and fixed one-pass Semantic/Hybrid generation. Add separate no-rewrite and limited-read ablations rather than attributing all full-agent gains to tool selection. Keep final synthesis, answer schema, corpus, and accounting comparable; report quality/cost frontiers where identical budgets are impossible. The existing tool-selection plan addresses this gap but has not supplied results.

2. **Make expansion and evidence packing sensitive to the task.** Whole-document `read` and repeated observations can spend the evidence allowance on irrelevant or overlapping material. Test bounded section/neighbor expansion and exact-overlap packing against current full reads, with identical budgets and provenance. Measure bridge-span retention, grounded answers, and abstention. Reuse the separate generation layer’s evidence discipline where suitable; do not assume plugging it into the agent will improve quality.

3. **Evaluate completion and gap recognition beyond exact repetition.** Distinct snippets can still fail to answer the missing fact, so the reminder may never fire. Test an explicit answerable/remaining-gap decision under the same total request budget. Include ambiguous and unanswerable cases, and grade correctness and premature stopping. Rebenchmark the current reserved-finalization contract first; historical max-turn rates are not current rates.

4. **Improve reranking around evidence usefulness.** Long-query allocation no longer drops all body text, yet the technical-domain gap remains; P3 bridge loss occurred without truncation. Compare passage selection, chain-aware evidence selection, or carefully bounded alternate reranker inputs on frozen pools. Check SciFact preservation, technical aspects, bridge evidence, answers, and CPU cost. Longer input alone is not an evidenced solution.

5. **Test model/backend alternatives as separate configurations.** A larger agent may improve control; MLX may accelerate embedding. Neither has a current end-to-end quality/cost result. Use separate caches and matching query/document embeddings for changed precision/backend, and a fresh current-contract model matrix for agent sizing. Close vector cosine agreement is insufficient to declare equivalent rankings.

### Lower-priority ideas

Broader chunk-size/overlap sweeps, query-conditioned fusion weighting, ANN tuning, and faster BM25/live-document access are plausible after profiling and with span-level evaluation. The current evidence does not select their parameters. A graph layer, learned retrieval policy, multilingual expansion, or a wholesale retrieval rewrite has no demonstrated advantage for this implementation and should not precede the diagnosed candidate, delivery, and answer-evaluation problems. MCP is an integration opportunity, not an evidenced retrieval-quality optimization.

## 8. Current ARKB Mental Model

Think of ARKB as a researcher using a small, reliable library desk.

**The library desk** knows exactly which version of a document was indexed, how to locate text, how to rank chunks, and how to trace evidence back to its source. It offers literal matching, lexical relevance, semantic relevance, and rank fusion. It makes these actions inspectable and independently testable. It does not decide what the user needs next.

**The researcher** is the agent. It chooses whether to search, what wording to use, which mode to request, which document to open, and when to answer. It can use one discovery to guide another, which is the capability a fixed retrieve-once pipeline lacks. That flexibility can also produce needless searches, incomplete enumeration, poor expansion, unsupported direct answers, or failure to finish.

**The execution contract** prevents common mechanical failures from masquerading as success. Evidence references bind source/revision/span; bad calls can be corrected; collection closes before the last answer opportunity; errors retain partial traces. It guarantees neither that the evidence is sufficient nor that the model understands it.

**The evidence supports selective conclusions:** Hybrid helps on several tested domains and hurts FiQA; stable reranker ties fix a measured ranking failure, but repaired reranking still harms the tested technical domains; source-fusion alternatives were correctly rejected as shared replacements; snapshot/cache contracts and bounded upserts have verified operational value. A small experiment supports improved agent completion, and an older model matrix supports a capacity effect.

**The experimental frontier is the control-and-answer layer.** The repository has not yet shown, with independently reviewed answers and comparable costs, how much value current tool selection, reformulation, reading, and stopping add over fixed RAG. The next useful work is to preserve needed evidence through each stage and measure whether the resulting answer is actually correct and supported.

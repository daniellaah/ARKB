# Phase C full-corpus indexing allocation

The complete BrowseComp-Plus corpus produces 1,883,193 chunks and 1,847,403
unique embedding inputs using the unchanged production chunker and tokenizer.
The corpus, queries and both official label sets remain unchanged.

The Phase B Qdrant writer first converts **all** vectors into Python float lists
and PointStruct objects, then sends slices of 128 points. On this CPython runtime,
a 1,024-dimensional Python float list occupies 32,824 bytes. At the measured
corpus size, vector lists alone would occupy 57.568706 GiB, in addition to the
14.367622 GiB float64 matrix, validation copies, source records, input corpus,
payloads and Qdrant Server. This is an avoidable temporary allocation in the
full-corpus indexing path, independent of query relevance or fusion scores.

The bounded refactor constructs PointStruct objects inside the existing 128-point
write loop. It preserves complete record/vector validation before the first
request, identical point IDs, vector values, payloads, batch sizes, batch order,
collection name and `wait=True`. It changes no Qdrant configuration, embeddings,
parser, chunking, document identity, index publication, retrieval or freshness
semantics. The normal builder still verifies the entire remote snapshot before
publication. No benchmark-specific condition is added to production code.

This is the necessary cross-cutting change allowed by Phase C Task 28 and is
reported separately from the frozen fusion decision. C0 is retained unchanged;
NFCorpus/FiQA results are not used to tune any algorithm. Their original measured
source copies remain preserved. BrowseComp uses a fresh measured-source copy and
records this allocation-only difference.

`evaluation/audits/audit_phase_c_upsert.py` compares the public upsert operation
against source loaded from accepted Phase B commit `75ba733`. The deterministic
fixture has 4,097 records and 1,024-dimensional vectors, covering 32 full requests
and one partial request. A hashing transport verifies the exact serialized
request stream without retaining vectors or calling a model/server. An invalid
final vector must fail before any write. Tracemalloc records temporary allocation
under the same fixture; its timed calls include tracing and serialization and
are not production latency benchmarks. Before/after outputs are versioned.

Full deterministic, service and freshness verification must pass again after the
refactor. Earlier successful verification is preserved as pre-refactor evidence;
overlapping suites and repeated runs are not added to inflate test counts.

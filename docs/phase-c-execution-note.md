# Phase C embedding throughput and interrupted-worker recovery

The full BrowseComp corpus has 100,195 documents, 1,883,193 chunks and
1,847,403 unique embedding inputs. Input deduplication therefore removes only
35,790 repeated inputs (1.9%). The corpus contains approximately 3.30 GB of raw
text data; 263 documents have at least one million body characters. The full
corpus and production 512/64 chunking remain unchanged.

Before the interruption, checkpoint differences measured 17.22 inputs/second
(33,010 to 50,003 inputs over 986.945 seconds). The last 200 successful embed
requests in the Ollama log averaged 17.905 inputs, 7,974.21 tokens and
1.007821 seconds per request, equivalent to 17.766 inputs/second inside the
model service. These are observations from the existing run, not new model
calls or a controlled performance benchmark. The sampled input length averaged
445.36 tokens. Different windows and request/checkpoint overhead preclude
assigning an exact percentage of time to each component.

The client caps requests at 32 inputs and 8,192 total tokens, sends requests
synchronously, validates the returned vectors and commits each batch before
requesting the next. The running model was Qwen3-Embedding-0.6B Q8_0, all 29
layers offloaded to the Apple M4 Max Metal GPU. The server used `-np 1`, and its
log showed inputs processed through a single slot. Model residency on the GPU
does not establish full GPU utilization or efficient parallel batch execution.
The close service and end-to-end throughput measurements point to the current
inference path as the main steady-state cost. Qdrant is not involved in this
cache-only phase; the earlier bounded-upsert refactor reduces later indexing
allocation and does not accelerate embedding inference.

At the observed rate, full embedding takes approximately 29–30 hours, excluding
initial corpus scanning/chunking/token validation and subsequent index building,
retrieval and archiving. Throughput may vary with input length. The prior
remaining-time estimate assumed uninterrupted execution. A performance change
would need a separate small benchmark preserving the exact model, input text,
token handling, vector identity requirements and cache semantics; a larger HTTP
batch alone is not evidence of faster GPU execution. No concurrency or model
configuration was changed in this recovery.

At 2026-09-12 04:12 UTC, all three prior tool-session workers were absent and
their sessions were unavailable. The last progress checkpoint was at 04:09:58
UTC. Ollama logged an interrupted final request at 04:10:55 UTC (HTTP 400), but
the Python logs had no traceback or final status. The exact cause of worker
termination is undetermined; the stale `embedding`/`running` status files must
not be interpreted as process liveness.

SQLite quick-check and every durable vector's specification, checksum, length,
finite values and unit norm passed: **50,970 cached vectors** are reusable.
The interrupted database, WAL, status files and logs were preserved with hashes
under `/Volumes/ARKBPhaseC/recovery-20260912T0420` before reopening the cache.
The unchanged frozen helper reconstructs its input plan and then skips all
previously cached inputs. Input-plan reconstruction adds preparation time but
does not regenerate the saved vectors. The recovered plan must match the
original input hash, corpus/chunk counts and model/tokenizer configuration
before validation is allowed to begin.

Recovery runs as one OS job, `com.arkb.phase-c.recovery-20260912-0420`, with
`RunAtLoad=true`, `KeepAlive=false` and no calendar or repeat interval. A
process-scoped `caffeinate -i` assertion prevents idle sleep while it runs.
The job is independent of the interactive tool session. It does not restart
automatically after failure or guarantee survival across logout/reboot or disk
removal. Its status and per-stage logs are in the recovery directory.

After embedding, the validation driver verifies the completed FiQA capture
against its prior checksums and skips it. An existing unfinished or changed
capture is rejected. Four operational guard checks passed without any model or
retrieval calls; they are separate from the product regression counts. The
remaining BrowseComp capture still runs once, followed by offline replay,
snapshot preservation and final verification. No architecture tuning or Agent
evaluation is introduced.

## Small parallelism probe

The installed Ollama 0.33.2 scheduler explicitly forces embedding-only models
to `numParallel=1`, regardless of `OLLAMA_NUM_PARALLEL`:
[version-pinned scheduler](https://github.com/ollama/ollama/blob/v0.33.2/server/sched.go#L469-L475).
Its HTTP embedding handler already dispatches the items in one input array
concurrently toward that runner. Increasing client concurrency alone therefore
does not enable multiple inference slots for this model.

A separate throughput probe used the first 256 saved FiQA corpus chunks whose
complete prepared inputs contained 400–512 tokens (searching only the first
5,000 stored ordinals). It used no queries or relevance labels. All three arms
used identical inputs and production request limits, model digest, 8,192-token
context, tokenizer and vector representation. Both servers were warmed before
timing; three repetitions rotated the arm order. BCP CPU preparation continued
during the probe. Model inference plus client request handling was timed;
cache writes and model loading were excluded.

| Configuration | Mean inputs/second | Throughput gain vs serial |
| --- | ---: | ---: |
| One client, one Ollama instance | 16.2796 | — |
| Two clients, one Ollama instance | 16.7311 | 2.8% |
| Two clients, two independent Ollama instances | 19.0172 | 16.8% |

All 2,304 timed vectors across nine arm/runs matched the saved cache and serial
reference **exactly**, with maximum absolute difference zero. Vector stream
SHA-256 was `a58b61f3d3f60bfea538af3e053da499d0047bd920e7b4a9a04db5dea5e197c7`
for every arm/run. The original FiQA SQLite checksum remained unchanged. The
temporary second Ollama instance was stopped after the probe.

Two independent instances can generate inputs concurrently, but they share the
same GPU. On this sample, 16.8% higher throughput corresponds to 14.4% less
inference time. It does not establish full-BrowseComp speedup or universal
floating-point equivalence on untested inputs. No full job, production cache,
retrieval policy or evaluation configuration was changed by this probe. A
production parallel writer would need bounded outstanding batches, stable
input-to-vector mapping and one ordered SQLite writer, while retaining durable
checkpoints and the original model/input identity.

The [recorded results](../evaluation/phase-c/v1/embedding-parallel-audit.json)
and [artifact manifest](../evaluation/phase-c/v1/embedding-parallel-artifacts.json)
preserve the exact sample, nine timing observations, script, replica log and
hashes. This embedding-only operational probe is not another retrieval run or
architecture-selection experiment.

## CPU preparation and direct slot probes

A one-second stack sample of the recovered serial worker showed tokenizer
encoding and pre-tokenization while it was using approximately one CPU core.
The job was still preparing inputs, with no ongoing embedding requests. That
explains GPU inactivity at this stage; GPU residency and utilization remain
different measurements.

A separate corpus-only probe selected 26 BrowseComp documents by file size and
filename order, including two documents over one million characters. Their
3,801,340 body characters produced 2,096 chunks. The unchanged public chunker,
tokenizer and document preparation ran in three configurations:

| Preparation | Seconds | Equality to original |
| --- | ---: | --- |
| Original serial | 12.2880 | Reference |
| Serial with per-document token-count memoization | 11.2437 | Exact |
| Eight processes with per-document memoization | 4.4190 | Exact |

Every chunk dataclass field, prepared input and token count matched. The
eight-process sample was 2.78 times as fast. This is one timing per arm,
including pool startup and result serialization but excluding corpus scanning;
it does not establish full-corpus or end-to-end acceleration.

Direct inference feasibility was also tested using the installed Ollama
`llama-server` binary and the same Q8_0 GGUF, on 128 of the fixed FiQA inputs.
An initial attempt completed its one-slot arm but failed on port reuse before
the eight-slot arm; its partial results are preserved. The second attempt used
separate ports and disabled prompt caching. Two one-slot runs achieved 17.0526
and 17.0630 inputs/second; eight-slot runs achieved 16.7715 and 17.0144. Logs
confirmed simultaneous slot processing, but there was no throughput gain.

Both direct configurations differed numerically from stored vectors. Maximum
absolute difference in the second attempt was 0.000186311. The direct client
used a Python emulation of Ollama's final float32 normalization and JSON
conversion; its exact equivalence is unproven. First-pass/later-pass differences
persisted with prompt caching disabled, so caching is **not an established
cause**. Warmup, numerical execution and normalization/serialization were not
isolated. No direct-backend output was adopted. All temporary servers stopped,
and the original FiQA SQLite checksum remained unchanged. No GPU utilization
percentage was measured by these probes.

## Resumable full input preparation

At 2026-09-12 05:11 UTC, a separate one-shot job started eight-process full-corpus
preparation on the approved external disk. It uses the unchanged public parser,
chunker, tokenizer and input preparation. Source hashes match the original
frozen helper's production modules. Each document has an atomic compressed
input shard and checksum receipt covering both text and token counts. Resume
reuses only verified shards with matching document revisions; uncommitted
partial writes are recomputed. A deterministic merge restores the original
source/chunk order and deduplicates inputs by first occurrence.

A completed plan is published only if all 100,195 documents, 1,883,193 chunks,
1,847,403 unique inputs and the original full ordered input SHA-256 match:
`76ff6e3dcc329c6a7ddd3490f0c36587ed813e0e667f007dc62b7a58ed5d70ab`.
The consumer verifies the manifest, original checkpoint, model/tokenizer/source
identity and input-file checksum before opening a cache writer. Inference
retains the original model, request limits, vector representation, retries and
single ordered SQLite writer. The saved 50,970 vectors remain reusable. The
normal final index builder still uses its production preparation path; this
optimization is confined to operational cache staging and recovery.

Eight operational checks passed on a temporary 27-document corpus, including
a duplicate document: 2,109 chunks and 2,096 unique inputs matched the serial
reference in order, metadata and token counts. Corrupt counts, changed revisions,
changed inputs and tokenizer drift were rejected. Four additional workflow
checks used fake processes/services and temporary databases to verify the
cutover gate, PID guard, preservation and refusal to interrupt a job that has
already begun embedding. These 12 checks make no model/retrieval calls or
production-cache writes and are separate from the product regression counts.

The original serial recovery remains running until full-plan verification.
Only the recorded original job/PIDs may then be stopped, and only while still
in preparation. The new workflow waits for those processes to disappear,
preserves a SQLite backup and prior status/logs, then consumes the verified
plan and resumes the registered validation/finalization sequence. If the
original job has already advanced to embedding or beyond, it is left running
and the prepared plan is retained for future recovery. Any verification failure
stops the new job; it does not silently change inputs or retry retrieval.

Job: `com.arkb.phase-c.prepared-20260912-0512`; status and stage logs:
`/Volumes/ARKBPhaseC/preparation-20260912T0512`. Full-plan progress:
`/Volumes/ARKBPhaseC/plans/browsecomp-plus-v1/status.json`.
At the time this note was recorded, full preparation and cutover were still
pending. Both one-shot jobs have no calendar, repeat interval or automatic
restart. The CPU optimization does not establish an improvement to the
approximately 29–30 hours of embedding work at the previously observed rate.

See [CPU probe](../evaluation/phase-c/v1/embedding-preparation-audit.json),
[direct-slot probe](../evaluation/phase-c/v1/embedding-direct-slots-audit.json),
[input-plan checks](../evaluation/phase-c/v1/input-plan-audit.json),
[workflow checks](../evaluation/phase-c/v1/prepared-workflow-audit.json) and the
[artifact manifest](../evaluation/phase-c/v1/embedding-preparation-artifacts.json).

## Subsequent GPU backend investigation

The user's follow-up GPU acceleration request led to isolated TEI Metal and
native MLX float16 probes on the same fixed 256 FiQA inputs. MLX batching
measured 25.0245 inputs/second versus 16.4693 for Ollama in its paired probe
(51.9% higher throughput, 34.2% less inference time). TEI Metal was slower.
Both alternatives differ from the original Q8_0 vectors and were not adopted.
The original full Phase C job and cache remain unchanged. See the
[throughput investigation](embedding-throughput-investigation.md) for methods,
precision differences, GPU activity readings and the required separation from
the frozen retrieval evaluation.

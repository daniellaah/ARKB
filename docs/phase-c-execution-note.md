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

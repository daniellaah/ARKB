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

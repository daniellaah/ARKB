# Phase C index preparation reuse

The full BrowseComp-Plus embedding cache completed on 2026-09-13. The validation
builder then repeated serial Markdown chunking and token counting because the
earlier parallel input plan had only been connected to the embedding cache
writer. A two-second macOS sample at 07:41 PDT observed 169 of 179 main-thread
samples in tokenizer encoding. The SQLite candidate-build table was still empty.

The user authorized trying an equivalent preparation path before deciding whether
to replace the running preflight. This changes no corpus, model, chunking budget,
vector representation, fusion policy, retrieval depth, query population or label.
The frozen Phase C development decision remains C0.

## Recover and verify complete chunk metadata

`evaluation/experiments/recover_phase_c_chunks.py` reads the existing prepared
document shards. Each shard contains the original document revision, ordered
embedding inputs, validated token counts and a hash of **all original chunk
fields**, including source coordinates, section metadata and occurrence numbers.
The original plan manifest anchors the inventory and each compressed shard hash.

The recovery locates candidate verbatim slices and uses the unchanged production
Markdown section/metadata functions. It accepts a document only when the complete
reconstructed chunk hash equals the saved original hash. Ambiguous repeated text
falls back to the original chunker and matching local tokenizer, with the same
512/64 budgets. Fallback output must pass the same hash and input comparisons.
Eight CPU processes operate on independent documents; the final inventory restores
the original source order. Completed recovered shards have checksum receipts.
No model calls or index writes occur during staging.

The initial 26-document, 2,096-chunk sample recovered in 0.402 seconds, excluding
initial plan/tokenizer/inventory checks. This is a warm, small operational probe,
not an estimate of full index construction or a Qdrant throughput measurement.
Full-corpus results are recorded in the external recovered plan manifest.

## Use the artifact with the original index builder

`evaluation/experiments/use_phase_c_chunks.py` temporarily supplies the original
builder with checked chunks and token counts. Before any candidate is created it
checks every original and recovered shard, current document revision, complete
chunk metadata, original token counts, input ordering and full original input
digest. Each subsequent input validation must match its expected text, source,
ordinal, tokenizer and active token limit. The immutable Note revision is computed
once per loaded document and reused during this build; the original record
constructor still checks each source slice.

The original SQLite/cache validation, Qdrant writer, complete remote verification
and READY publication all run unchanged. Temporary bindings are restored before
retrieval, including on exceptions. Stage observers report chunk loading,
candidate creation, SQLite records, snapshot loading, Qdrant transfer/verification
and publication. They forward the original calls unchanged.

`evaluation/audits/audit_phase_c_chunk_recovery.py` compares complete original and
prepared builds using the same 26 documents and cached deterministic 4-dimensional
fixture vectors. The vector service is a recording test double. READY manifests,
cache counts, ordered records, vector bytes and remote call sequence must match.
It also covers source/plan changes, wrong scope/order/budgets/input text, empty
bodies, Markdown/code sections, repeated text and binding restoration. Its timings
exclude real model inference and real Qdrant indexing and are not a production
speedup claim. See `evaluation/phase-c/v1/chunk-recovery-audit.json`.

## Conditional one-shot handoff

`evaluation/experiments/resume_phase_c_chunks_once.py` requires a completed full
recovered plan and matching adapter audit. It checks the exact launchd job and
four-process parent chain, then briefly pauses only the checked worker to confirm
that no candidate, snapshot, active index or retrieval capture has been created.
If the original job has already advanced, it resumes/leaves that job running and
does not start another writer.

If the gate passes, it stops the identified job, verifies that its processes have
exited and preserves the complete prior validation directory. A SQLite backup
copies all committed embedding-cache pages, including WAL contents; no vectors
are regenerated. The new validation records its operational adapter, recovered
plan, prior attempt, audit and source hashes in its protocol. It keeps the original
measured source, original retrieval capture/scoring code, offline replay and finalizer.
It never retries a retrieval capture. Recovery failure is explicit and stops the
new workflow.

`evaluation/audits/audit_phase_c_chunk_cutover.py` exercises six fake-process
fixtures: invalid plan, already advanced build, changed PID/parent identity,
advancement at the pause gate, incomplete copied cache, and successful sequential
handoff through validation/replay/finalization. These tests mutate no real process,
model, vector service or production artifact. See
`evaluation/phase-c/v1/chunk-cutover-audit.json`.

The interrupted preflight time and recovery time remain setup costs, separate from
the single retrieval capture timings. The stopped attempt is preserved as an
unscored operational attempt. Agentic evaluation continues to require the READY
index, complete Phase C validation/replay and final archive; staging completion
alone does not satisfy that gate.

## Verified adoption on 2026-09-13

Full recovery completed in 927.871 seconds (15 minutes 28 seconds): all 100,195
documents and 1,883,193 chunks matched their original complete metadata hashes
and embedding inputs. There were 129 original-chunker fallback documents and
100,066 documents recovered without re-tokenizing. No model or index calls were
made by the recovery stage. Its manifest is preserved in
`evaluation/phase-c/v1/index-preparation-reuse/full-recovery-manifest.json`.

The handoff gate passed at approximately 08:18 PDT: the original worker still had
zero candidate builds, snapshot rows, active indexes and retrieval captures. The
four original workflow processes exited. Their validation directory was moved
intact to `/Volumes/ARKBPhaseC/index-recovery-20260913T0820/prior-validation-run`.
The new one-shot job is `com.arkb.phase-c.chunks-20260913-0820`, with status at
`/Volumes/ARKBPhaseC/index-recovery-20260913T0820/status.json`. Its filename uses an
operation label; exact execution timestamps are in that status and the cutover gate.

The half-hour Agentic heartbeat was updated to follow the new workflow and to
leave the superseded job stopped. This adoption proves complete preparation
equivalence and a successful guarded handoff; it does not declare the full
Qdrant index, Phase C validation or Agentic experiment complete.

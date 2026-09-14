# Agentic tool-selection execution

## Execution hold — 2026-09-14 UTC

Core-v1 and its heartbeat are paused for [design review](agentic-tool-selection-design-review.md).
Eleven attempts completed, one was administratively interrupted, and 5,448 were
not started. Preserve them as incomplete historical diagnostics; do not resume
the old schedule or treat unexecuted work as model failures. The dispatch notes
below describe history, not current readiness. Phase C remains complete.
A replacement requires revised tool-level gates and a new protocol.

## Historical core dispatch

The suspended execution is the **5,460-attempt core run**, registered in the
[core protocol](agentic-tool-selection-core-protocol.md). It started at
2026-09-14T04:08:34Z under job `com.arkb.agentic-tools.core-v1`. Its protocol hash is
`453906858f404cb5f4b2f0fa1ba919ead2a925ec182e4b7a5ac69131fba956d5`;
status is `/Volumes/ARKBPhaseC/agentic-tools-v1/core-v1/core-status.json`.
The core snapshot contains 90 files; 67 inference files match v2 exactly, and
46 boundary/scoring/statistics checks passed. The completed v2 pilot projected
49.67 hours of core inference, plus scoring and archival. Calibration's automatic
portion and 42 blinded exports completed; independent human review is pending.

The preceding complete replacement pilot **v2** is described in the
[registered amendment](agentic-tool-selection-pilot-amendment-v2.md). The first
pilot below is preserved as historical evidence. It completed 98/98 attempts
but exposed runtime and final-output failures; no core trials followed it.

The replacement uses the same Qwen3.5-4B weights with `think=False` in all seven
arms, an explicit ripgrep-capable background PATH and pretrial dependency checks.
Its executable protocol is
`185bdf30dc47f42b54d5d65ac4bc0c4158686938894c7d8463dfd915a700cc5b`,
with 81 frozen source files. Job `com.arkb.agentic-tools.pilot-v2` started at
2026-09-14T02:38:53Z; current status is in
`/Volumes/ARKBPhaseC/agentic-tools-v1/pilot-v2/pilot-status.json`.
The existing 30-minute heartbeat now explicitly follows this revision and its
review gates. All original attempts, corpus selections and READY indexes remain
available. The original one-shot pilot job has been unloaded after completion.

The conditional execution authorized in the experiment plan is underway. Phase C
completed on 2026-09-13 at 17:18 America/Los_Angeles, including all three broader
validation replays and preserved indexes. The unrelated local MLX evaluation
then exited; the user confirmed the GPU was idle before Agentic preparation.

## Frozen pilot

- Full BrowseComp-Plus, FiQA and NFCorpus indexes are reused read-only. FiQA keeps
  the previously registered 38-empty-document indexing exception and original qrels.
- Selection contains 14 pilot scenarios and 260 core scenarios. MuSiQue original
  pairs are kept together and stratified by hop count. All 66 required variant
  contexts have independent READY indexes prepared before trial timing.
- Seven arms, 98 pilot attempts, no reranking, the same 8,000-token evidence
  ceiling, and the planned model/tool/time ceilings are recorded in the design.
- The unchanged product Agent loop is bound to private evaluation instructions
  and capability definitions. Production source files have not been modified.
- 32 synthetic contract checks passed. They cover tool/mode escapes, missing and
  null defaults, reference restrictions, fixed-RAG prefix packing, withheld
  evidence, recorded model failures, finalization, pairing and attempt identity.
- The effective Qwen3.5-4B model digest is
  `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`.
  Provider preflight verified a 32,768-token context and canonical structured
  output. A synthetic oversized input with truncation disabled returned HTTP 400.

The pilot executable protocol SHA-256 is
`c8c0a5842769e1067ad989d1b363ec5ae0484d15e021133a58a388a86266da49`.
Its source snapshot contains 77 files. New offline analysis is separately hashed;
the running source snapshot is immutable.

## Context and grading details

Ollama 0.33.2 supports `truncate=false` and `shift=false` on chat requests. The
evaluation records and transmits both explicitly through a local HTTP adapter,
then parses the response using the existing Ollama response type. This prevents
automatic message removal and context sliding from silently changing evidence.
Temperature, thinking, context and output settings are shared by all arms.
The source of that behavior is retained under the external provenance directory.

The scoring-only importer verified 830 official BrowseComp answers against the
same pinned encrypted query source. Answers never enter inference scenarios or
indexed context files. The official rubric is extracted as a constant from
BrowseComp-Plus revision `046949032b0328319cc9a02663a759ec601d9402`; the remote
evaluation program is not imported or executed.

The grader is Qwen3-32B Q8_0 via Ollama. This is an **ARKB adaptation**, using
quantized GGUF weights and a different runtime from the official vLLM setup.
The official rubric, temperature 0.7, top-p 0.8, top-k 20, 4,096 output tokens and
non-thinking mode are retained. The parser requires one unambiguous correctness
field; absent/conflicting fields remain pending. Independent human calibration
and citation review are pending. No result is described as human verified.

## Execution and continuation

The one-shot local job is `com.arkb.agentic-tools.pilot-v1`. It has `KeepAlive=false`;
an exclusive file lock prevents a second inference worker. Completed attempts
have immutable identity and result/provider checksums. Incomplete attempts are
retained for explicit recovery accounting, not silently rerun.

The approved external output is `/Volumes/ARKBPhaseC/agentic-tools-v1`. Relevant files:

- `pilot-status.json`, `pilot.log`, `pilot-error.log`: current worker stage.
- `protocol.json`, `selection.json`, `inputs.json`, `context-indexes.json`: frozen inputs.
- `pilot-attempts/<key>/`: requests, responses, canonical outcome and checksums.
- `scoring/`: separate answer/support labels.
- `provider-preflight.json`, `judge-design.json`, `provenance/`: configuration evidence.

Small versioned artifacts are in `evaluation/agentic-tools/v1/`. The existing
30-minute heartbeat consults the orchestration record and the live pilot worker;
it must not recreate the pilot or restart completed Phase C jobs. After all
98 pilot attempts, run offline accounting, review technical failures, publish
the measured cost estimate and freeze a separate core protocol before dispatching
5,460 core trajectories. The current pilot worker deliberately cannot launch core
inference merely because its own loop ends.

No core or judge benchmark attempts have run at the time of this execution note.
Preparation and synthetic provider checks are excluded from measured trials.

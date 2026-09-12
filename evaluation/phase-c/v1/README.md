# Phase C v1 evidence

Development selection is frozen in `development-decision.json` at commit
`5bd06c6`: retain C0. `protocol.json` records the architecture/preregistration
hash before C1/C2 execution. `development-summary.json` includes full metrics,
paired bootstrap intervals, win/tie/loss counts, timing and the failed C3 gate.
`c0-diagnostics.json` records exact accepted-baseline reproduction and candidate
availability before alternative design. The reference architecture note is
`docs/phase-c-plan.md`.

Broader validation is in progress; do not interpret this directory as a completed
Phase C release. Initial service failures occurred before retrieval and are
retained in the working logs. No validation scores have informed policy design.

The full original validation corpora, official queries, evidence/gold qrels and
normalization manifests are preserved under `/Volumes/ARKBPhaseC/data`. This is
an APFS sparse disk image physically stored within the user-approved external
folder `/Volumes/闪迪1T/arkb-phase-c-v1`. Original hash-verified downloads remain
in that folder's `inputs` directory. Do not publicly publish decrypted
BrowseComp-Plus questions/answers; official source data carries an anti-leakage
canary. Local frozen retrieval artifacts are for reproducibility and Phase E.

Storage incident: Qdrant 1.19.0 on OrbStack host bind mounts reported incompatible
FUSE storage and timed out creating collections. A separate native Docker volume
was provisioned without changing Qdrant version or retrieval/index parameters.
See [Qdrant's official filesystem guidance](https://qdrant.tech/documentation/guides/common-errors/).
Existing services and historical READY indexes were not restarted or modified.

FiQA input audit found 38 documents with empty title and body; one is positive in
test qrels. Original corpus and labels remain intact. The user approved continuation with the explicit empty-document indexing exception.
Only 57,600 nonempty documents are indexed; all 648 queries and all labels remain unchanged.
See `empty-document-exception.json`. Never silently discard its
positive label or its affected query.

# Phase B v1 evidence

Completed 2026-09-11. [Engineering report](../../../docs/phase-b-report.md).

**Decision: reranking remains optional.** The registered rule selects B3:
512 total tokens, query head/tail cap 128 (including gap), title cap 64,
remaining body with a 128-token floor when available, stable ties, depth 20,
and explicit deterministic fallbacks. The model and all upstream retrieval
remain unchanged. These are public development results, not release validation.

| Dataset | Hybrid nDCG@10 | Corrected B3/B5 |
| --- | ---: | ---: |
| SciFact (300 queries) | 0.721640 | 0.766586 |
| Stack Overflow (115 queries) | 0.480653 | 0.407530 |
| Robotics (101 queries) | 0.390414 | 0.355810 |

The [summary](summary.json) contains full precision, all metrics, paired intervals,
wins/ties/losses and diagnostics. [Decision](decision.json) includes the shared
selection rule result and controlled paired contrasts; [protocol](protocol.json)
was registered before B3/B4 inference. [Scope audit](scope-audit.json) records the
three changed production files and the accepted Phase A baseline.

## Portable archive

`experiments.tar.gz.part-000` through `part-003` are lossless 32 MiB transport
parts (the final part is smaller) of a 131,290,615-byte gzip archive.
[parts.json](parts.json) seals their order, sizes and SHA-256 hashes.
[archive-manifest.json](archive-manifest.json) lists all 620 contained evidence
files and the original archive hash. The archive root is `phase-b-v1/`.

The experiment archive includes:

- Exact frozen candidate pools and selected chunks; original Hybrid top 100,
  component ranks/scores, text, coordinates and source/revision identities.
- Separate scoring-side qrels/aspects; dataset manifests, cards and supplied license.
- All B0/B3/B4 real inference scores, actual input IDs, rankings and wall times;
  B1/B2 score replays; B5 final production input and score replay.
- Original, measured and final source snapshots, final test source, tokenizer
  files, pinned model file hashes, lockfile and recorded software environment.
- Initial and corrected instrumentation attempts; all earlier analyses; final
  B0–B5 analyses with per-query metrics, attribution, strata and checksums.
- Registered policy selection, input-loss and tie diagnostics, post-hoc examples,
  and supporting scripts. No original P4 or prior Phase B result was overwritten.

Model weights and upstream SQLite/Qdrant indexes are not needed for score replay
and are not bundled. Fresh inference requires the recorded pinned weights and
measured environment. The historical `p4-provenance/*/checksums.json` files refer
to the original P4 runs, including their external indexes; the independent Phase B
archive manifest verifies every file actually included here.

B5 uses the completed B3 model scores, validates all 10,320 prepared inputs and
516 rankings through final production code, and re-scores three fixed queries
(60 candidate scores, all exact). It is not a second full inference trial.
All-equal and all-body-empty triggers occur zero times under B3; B5 is therefore
quality-identical. B5b adds no separate quality arm, and the registered B6 gate fails.

## Offline verification

With the project's `evaluation` dependencies installed, from the repository root:

```sh
.venv/bin/python evaluation/audits/replay_phase_b_archive.py \
  evaluation/phase-b/v1/parts.json \
  --output /tmp/arkb-phase-b-replay-new.json
```

Choose a new output filename. This verifies all parts, reconstructs the exact
archive, checks every extracted file, then uses the archived production source
and archived scoring scripts to reproduce all six final analysis summaries,
the policy selection and the complete frozen Hybrid boundary. It makes **zero
model calls and zero upstream retrieval calls**. Temporary extracted data is
removed after verification. [replay-validation.json](replay-validation.json)
records the successful byte-identical replay of the delivered archive.

The first packaging verification encountered macOS `/var` versus `/private/var`
path aliases. The failing member's content hash already matched. Resolving the
temporary root before checking containment fixed the verification harness;
no experiment, model score or archive byte changed. The initial log is retained.

## Verification evidence

[verification/test-summary.json](verification/test-summary.json) reports:

| Group | Passed | Failed / errors | Skipped |
| --- | ---: | ---: | ---: |
| Deterministic suite | 1152 | 0 | 0 |
| Service integration | 60 | 0 | 0 |
| Evaluation subset | 248 | 0 | 0 |
| Phase A subset | 378 | 0 | 0 |
| Reranker deterministic subset | 48 | 0 | 0 |
| New reranker deterministic cases | 38 | 0 | 0 |
| Reranker real-model subset | 5 | 0 | 0 |
| New real long-query case | 1 | 0 | 0 |
| Live freshness checks | 16 | 0 | 0 |

Subset rows overlap. Full JUnit, raw logs, red/green regression evidence and the
complete isolated freshness fixture are in `verification/`, separately from the
sealed experiment archive. The final production commit is `75ba733`; no
production code changed during final verification or archive replay.
`bundle-checksums.json` seals the complete delivered directory, including these
verification files, the transport parts and replay result.

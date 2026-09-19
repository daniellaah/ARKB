# Registered studies (frozen history)

Each subpackage holds the one-off preparation, registration and pipeline
scripts of one registered study. They are kept runnable for replay and audit
but are not part of the reusable library in `evaluation/agentic_tools`
(capabilities contract, transport, readiness gates, stop controller, selection,
records, scoring, statistics, judge, analysis, runner, preflight, registration).
Raw artifacts live on the external volume under
`/Volumes/ARKBPhaseC/agentic-tools-v1/<run>`; small public copies live under
`evaluation/agentic-tools/<run>/`. Frozen runs carry their own source snapshot,
so moving these files does not alter any registered protocol hash.

| Study | Status | Scripts | Public record |
| --- | --- | --- | --- |
| `agentic_pilot_v1` | completed 98 attempts, superseded | prepare, freeze, probe_finalization | `evaluation/agentic-tools/v1`, `v2` |
| `agentic_core_v1` | stopped after 11 attempts (design review) | freeze_core | `evaluation/agentic-tools/core-v1`, `design-review-v1` |
| `agentic_pilot_v3` | completed 98 attempts, audited, tool readiness passed | prepare, pipeline | `evaluation/agentic-tools/v3` |
| `agentic_serving_v1` | completed 90 requests, negative concurrency screen | probe | `evaluation/agentic-tools/serving-v1` |
| `agentic_repeatability_v1` | completed 70 attempts, audited | repeatability | `evaluation/agentic-tools/repeatability-v1` |
| `agentic_core_v2` | registered, stopped after 16 attempts by user decision | design, pipeline | `evaluation/agentic-tools/core-v2` |

The judge, labels and scoring modules keep their calibration-time behavior; a
future registered study re-runs calibration and re-pins their hashes.

# Repaired Agentic pilot v3

The background pipeline was started at 2026-09-14 06:06 UTC and initially waited
for unrelated LoRA training. This start record does not claim benchmark inference
has begun. Read the live external status for subsequent progress.

- Output: `/Volumes/ARKBPhaseC/agentic-tools-v1/pilot-v3`
- Job: `com.arkb.agentic-tools.pilot-v3`
- Live pipeline: `pipeline-status.json`, `pipeline.log`, `pipeline-error.log`
- Registered run (created after provider preflight): `protocol.json`,
  `measured-source-manifest.json`, `pilot-status.json`, `pilot-attempts/`
- Operational checks: `scope-readiness/`, `scope-readiness-manifest.json`,
  `tool-readiness.json`; final accounting: `pilot-accounting.json`
- Local amendment: `docs/agentic-tool-selection-pilot-amendment-v3.md`
- Durable handoff: `evaluation/results/agentic-tools-v1/orchestration-status.json`

All 1,270 selected regression tests passed, with 60 integration-marked cases
excluded; see `boundary-tests.xml`. This includes 19 operational gate, resource
ownership and graceful-stop checks. Replaying the new gate on the old v2 records
rejected all 25 timeouts and the slow BrowseComp BM25/Hybrid tool groups, despite
schema-valid final answers. No historical experiment artifact was modified.

`registration-inputs.json` pins all 96 source/configuration files and 20 prepared
inputs to the tested version while resource availability is pending. The pipeline
refuses registration if that inventory or content changes. Once registered, it
executes from the frozen source and retains the root shared inference lock path.

The pipeline checks resource availability every 30 seconds while waiting, without
holding the shared inference lock. The existing Codex heartbeat follows up every
30 minutes; it must not launch a duplicate job. The pipeline only runs this pilot
and its accounting. Concurrency/repeatability measurements and the replacement
core design still follow; no automatic transition to the old 5,460-attempt study.

For an administrative pause, create `stop-request.json` in the output directory
or send SIGINT/SIGTERM directly to the Python pipeline/worker PID. The current
trajectory finishes and is saved before pausing. A forced kill or launchd service
removal can still interrupt a trajectory; prefer the stop file. Do not remove a
stop file or retry an operational failure without reviewing its cause. Existing
completed attempts are immutable and are not repeated after an authorized resume.

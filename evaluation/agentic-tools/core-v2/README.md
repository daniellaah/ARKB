# Replacement core v2 (administratively stopped after 16 attempts)

Stopped 2026-09-18 UTC by user decision to fix diagnosed product problems first.
The run directory keeps `stop-request.json`, `administrative-stop-accounting.json`,
`core-accounting.json` (partial, 16 attempts) and one archived GPU-competition
pause with its review. Do not resume or bootstrap the job again.

Registered design and public copies for the replacement core study described in
[docs/agentic-core-v2-protocol.md](../../../docs/agentic-core-v2-protocol.md).
The authoritative run lives at `/Volumes/ARKBPhaseC/agentic-tools-v1/core-v2`.

- `precision-cost.json`, `selection.json`, `core-schedule.json`: outcome-blind
  design produced by `evaluation.agentic_tools.core_design` from the audited v3
  pilot and the exposed core-v1 question list (3,640 attempts: 3,360 primary
  single sessions plus a 280-attempt registered BrowseComp repeat subset).
- `draft-v0-20260914T0907/`: the earlier 3,360-attempt draft without the repeat
  subset, kept for the record; it was never registered.
- `tests.xml`: the regression run pinned at preparation (1,296 passed, 60
  integration-marked cases deselected).
- `core-v2.plist`: the one-shot launchd job `com.arkb.agentic-tools.core-v2`. It
  polls every 30 seconds without holding the shared inference lock, registers the
  protocol only when no competing GPU process is present, then executes from the
  frozen source and continues with accounting, the independent audit, grading
  and analysis. Bootstrap it once; never start a second copy.
- `protocol.json`, `core-accounting.json`, `completion-audit.json`,
  `grading-summary.json` and `analysis-*.json` appear here as the pipeline
  reaches each stage.

Administrative pause: create `stop-request.json` in the run directory or send
SIGINT/SIGTERM to the worker; the current trajectory finishes and is preserved.
Registered pause rules and the 32-hour trajectory time cap are in the protocol.
Review a pause before any resume; do not delete the stop file or retry started
trajectories. Human calibration and citation review remain pending, so every
answer-quality figure is provisional.

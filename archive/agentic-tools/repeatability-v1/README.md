# Full Agent repeatability v1

Completed: 70 new trajectories plus 35 pinned historical v3 trajectories.
The full immutable run lives at
`/Volumes/ARKBPhaseC/agentic-tools-v1/repeatability-v1`.

See [results](../../../docs/agentic-repeatability-v1-results.md) and
[completion-audit.json](completion-audit.json). Local result files are copies of
the frozen run's summaries; they do not replace the raw provider journals.
The one-shot launchd job exited with code zero and was removed after audit.
Do not restart it or use its completion to resume the old core schedule.

# Repaired Agentic pilot v3: operational results

Completed 2026-09-14 06:51:42 UTC. Protocol SHA-256:
`8c3eb9ca245fd7912630b7e4d0e6900120a0b854c14d42dbd9c85856cbceeeac`.

All 98 trajectories across BrowseComp-Plus, FiQA, NFCorpus and MuSiQue completed.
The audit verified 96 source files, 435 actual provider requests and 1,211 trace
checks. All result/provider checksum records are intact. Of 384 executed tool
calls, 374 were valid and 10 were model input rejections. No valid tool call timed
out or failed operationally. Three final-output failures remain in the data: one
output-token limit and two invalid references. Answer quality has not been scored;
independent human review remains pending.

BrowseComp measured tool calls, excluding preparation:

| Tool | Calls | Mean seconds | P95 seconds |
| --- | ---: | ---: | ---: |
| match | 28 | 5.10 | 7.41 |
| BM25 | 14 | 0.72 | 1.81 |
| Semantic | 30 | 0.61 | 2.90 |
| Hybrid | 24 | 2.36 | 10.19 |

These are the repaired Agent's actual calls, not a fixed-query paired speedup
comparison against v2. The registered operational gate passed. Full corpus and
indexes were retained. Worker wall time was 40.04 minutes: measured trajectories
totaled 26.55 minutes and separately recorded setup totaled 10.65 minutes; scope
checks, verification and orchestration account for the remainder.

Applying these small pilot group means to the historical 5,460-attempt schedule
gives 27.79 inference hours, versus the old 49.67-hour projection. This is only a
capacity scenario, excludes setup/grading/review/archival, and is not a selected
replacement design or promised completion time.

## Auxiliary archive limitation

Nine final scope-check artifacts containing 90 fixture calls are independently
verified and passed. The runner prepared scopes 45 times because MuSiQue variants
alternate between contexts. Its scope-hash-plus-PID filenames overwrote earlier
checks on returning to the same context. Those earlier auxiliary timings cannot
be independently reconstructed. The frozen stop-on-failure control flow would
have paused on a failed check, but that is not a substitute for retained raw
measurements. This limitation does not affect the 98 immutable measured attempts
or their provider journals. The working runner now adds the setup sequence to
each artifact name; the historical v3 snapshot is unchanged.

The complete audit is in
`evaluation/agentic-tools/v3/completion-audit.json`, with the frozen accounting and
tool-readiness reports beside it. Next is the separately registered
[serving concurrency screen](agentic-serving-probe-v1.md), followed by full Agent
repeatability/throughput confirmation and a justified replacement core protocol.

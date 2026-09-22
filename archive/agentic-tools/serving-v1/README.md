# Local serving concurrency screen

The registered protocol is copied here; authoritative output is
`/Volumes/ARKBPhaseC/agentic-tools-v1/serving-v1`. Protocol SHA-256:
`3c2588b98800eae6178de036cc4634c19dc021ab8f4a4fdbb1980cd1510f5437`.

Ten actual requests selected by v3 schedule order are each repeated three times
at client concurrency 1, 2 and 4: 90 measured calls, plus nine synthetic warmups.
All model/options, selection, order, checks and the standalone runner were frozen
before execution. The runner imports the unchanged v3 source snapshot. This is
request-level serving and output repeatability, not full Agent throughput or
answer scoring. The complete plan is `docs/agentic-serving-probe-v1.md`.

Registration passed 1,276 regression tests; 60 integration-marked tests were
excluded. Source tests and operational stop behavior are included. Initial
dispatch was deferred because unrelated LoRA training was active, before any
measured request started. Consult the durable orchestration status for updates.

The saved `serving-v1.plist` starts the frozen probe with its exact environment.
Only bootstrap it once, after checking live GPU competitors, the shared lock,
and absence of an existing job/process. Job label:
`com.arkb.agentic-tools.serving-v1`. A started probe is not automatically retried.

Outputs include `status.json`, `execution.log`, `error.log`, per-call immutable
request/response records, `blocks-results.json`, `memory-samples.json` and
`summary.json`. Failed and administratively unstarted requests remain distinct.
The existing Codex heartbeat follows this work and advances to full Agent
confirmation only after reviewing all repetitions under the frozen screen rule.

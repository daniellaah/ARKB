# Tool overhead diagnostics

These are development diagnostics for the repair described in
`docs/tool-overhead-repair.md`. They do not replace the paused Agentic protocol,
Phase C results or either original pilot.

- `baseline-match/`, `baseline-bm25/`: full-corpus profiles using the immutable
  core-v1 source through PYTHONPATH, with preparation outside query profiles.
- `repaired-match/`: preserved intermediate separate-file cache profile. The
  final implementation subsequently introduced a packed literal store and
  parallel, bounded rg output. Do not report this intermediate directory as the
  final implementation's performance.
- `repaired-bm25/`: profiles after tie-key reuse and heap selection. Each complete
  response checksum matches its baseline. The BM25 source hash matches final code.
- `replay-v2/`: first complete 87-call replay, preserved before final boundary
  hardening. This run had zero errors, retained 62 prior successful results and
  checked 85 source references. All 25 previous timeouts completed.
- `replay-final/`: the same complete call set on final hardened source. Source
  manifests must match before and after; summaries record preparation, unprofiled
  probes, call errors, successful-evidence equality and live source checks.
- `regression-tests.xml`: scoped project/harness run, 1,251 passed and 60
  integration-marked cases deselected.
- `exact-equivalence.json`: 252 synthetic comparisons with the frozen original
  matcher, including empty scopes and zero-width regex behavior.

No raw benchmark questions, model-generated patterns, answers or returned excerpts
are copied into these public diagnostics. Calls and responses are represented by
digests. The three generic profiling queries are synthetic performance probes.

Reproduction (from the repository, using the existing Python environment):

```sh
PYTHONPATH=src .venv/bin/python evaluation/experiments/profile_tool_overhead.py --mode bm25 --output NEW_OUTPUT
PYTHONPATH=src .venv/bin/python evaluation/experiments/replay_match_overhead.py --pilot /Volumes/ARKBPhaseC/agentic-tools-v1/pilot-v2 --output NEW_REPLAY_OUTPUT
.venv/bin/python -m pytest -q tests evaluation/agentic_tools -m 'not integration'
```

Use a fresh output directory for each run. For the old profile, PYTHONPATH points
to `/Volumes/ARKBPhaseC/agentic-tools-v1/core-v1/measured-source/src`. Profiling and
raw wall times are different measurements. None of these programs generates
document embeddings, invokes an answer model, changes indexes or resumes a core
experiment. A cache preparation cost is always separate from warm query timing.

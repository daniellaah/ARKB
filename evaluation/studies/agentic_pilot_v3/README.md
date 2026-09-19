# Agentic pilot v3 (2026-09-14, completed and audited)

Repaired-tool rerun of the 98 development trajectories with frozen operational
gates. `prepare.py` copied the parent inputs and the readiness policy;
`pipeline.py` waited for resources, registered, executed and audited. Results:
[docs/agentic-pilot-v3-results.md](../../../docs/agentic-pilot-v3-results.md).

Run the pipeline with `python -m evaluation.studies.agentic_pilot_v3.pipeline`.

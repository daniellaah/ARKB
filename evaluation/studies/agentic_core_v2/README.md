# Replacement core v2 (2026-09-18, registered and stopped)

`design.py` produced the outcome-blind precision, allocation and cost plan
(3,640 attempts: 3,360 single sessions plus a 280-attempt repeat subset);
`pipeline.py` prepared, registered and executed it with pause rules and a
trajectory time cap. The run stopped after 16 attempts when the user chose to
fix diagnosed product problems first. Protocol:
[docs/agentic-core-v2-protocol.md](../../../docs/agentic-core-v2-protocol.md);
public record: `evaluation/agentic-tools/core-v2`.

Entry points: `python -m evaluation.studies.agentic_core_v2.design` and
`python -m evaluation.studies.agentic_core_v2.pipeline`.

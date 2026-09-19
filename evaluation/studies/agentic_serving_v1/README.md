# Serving concurrency screen v1 (2026-09-14, completed and audited)

`probe.py` replayed ten real requests at client concurrency 1, 2 and 4. No
level passed the 10% throughput rule; the qwen35 family has one execution
slot in Ollama 0.33.2. Results:
[docs/agentic-serving-v1-results.md](../../../docs/agentic-serving-v1-results.md).

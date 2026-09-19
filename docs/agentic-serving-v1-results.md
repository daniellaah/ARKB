# Serving-v1 results and backend diagnosis

Completed 2026-09-14 08:11:31 UTC. All 90 measured requests and nine synthetic
warmups passed immutable journal verification. No operational failures or
length-limited outputs occurred. Each of the ten requests produced the same
normalized message in all nine measured executions.

| Client concurrency | Median requests/second | Relative to one | Client call P95 |
| --- | ---: | ---: | ---: |
| 1 | 0.7085 | 1.0000 | 8.67 s |
| 2 | 0.7290 | 1.0289 | 5.62 s |
| 4 | 0.7284 | 1.0281 | 7.93 s |

Neither higher concurrency passes the preregistered 10% median-throughput screen.
All three blocks per level are retained. The first serial block took 38.97 seconds;
the other eight blocks took 13.65–14.12 seconds. Prompt-cache state is a material
limitation, so these repeated-request results do not establish throughput on a
core workload of unique questions or complete Agent trajectories. Three blocks
per level do not establish statistical significance or population determinism.

## Actual execution capacity

After execution, the live backend process was
`/Applications/Ollama.app/Contents/Resources/llama-server`, parented by Ollama,
with `-c 32768 -np 1`. The local server log also reports `n_seq_max = 1` for this
load. This was client concurrency over one actual model execution slot.

The exact installed release's scheduler places `qwen35` in the architectures
restricted to one parallel request, overriding larger requested parallelism.
Consequently, increasing OLLAMA_NUM_PARALLEL alone would not enable multiple
slots for this model in this release. See
[Ollama v0.33.2 scheduler](https://github.com/ollama/ollama/blob/v0.33.2/server/sched.go#L469-L480).
General parallel-request settings and their context-memory scaling are described
in the [official FAQ](https://docs.ollama.com/faq#how-does-ollama-handle-concurrent-requests).

No backend, model, global service setting or architecture safety constraint was
changed. The next bounded full-Agent repeatability study retains the same single
execution policy. Independent replicas or a different serving implementation
would need their own compatibility, resource and quality validation before
changing the registered comparison.

## Memory measurement limitation

The frozen process-name filter sampled the Ollama front process but missed its
`llama-server` child. The reported 63,776 KiB peak is therefore not a whole-model
memory measurement and must not be used for capacity planning. A subsequent live
snapshot showed the backend at 8,099,520 KiB RSS; this is a post-run observation,
not a reconstructed peak. The separately recorded API model allocation was
4,312,216,369 bytes of size_vram. RSS and this allocation describe different
quantities and must not be added as distinct physical memory consumption.

The working sampler now follows the server's process descendants and has a
regression fixture. Frozen artifacts remain unchanged. The completion audit is
`evaluation/agentic-tools/serving-v1/completion-audit.json`; its results are valid
for recorded request/throughput/output comparisons with the qualifications above.

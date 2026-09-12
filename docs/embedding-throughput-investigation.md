# Embedding throughput on the current Mac

MLX float16 batch inference reached **25.0245 inputs/second**, compared with
16.4693 for the original Ollama Q8_0 path on the same fixed sample: 51.9% more
throughput, equivalent to 34.2% less inference time. This is a prospective
backend/precision change, not a change applied to Phase C. Retrieval quality
was not evaluated, and the generated vectors differ from the production cache.

## Method and scope

The user requested acceleration of GPU embedding computation while full
BrowseComp input preparation continued. These are independent operational
probes with no queries, relevance labels, retrieval calls, production cache
writes or changes to the frozen fusion experiment.

Both probes reuse the first 256 saved FiQA chunks whose complete prepared input
has 400–512 tokens, selected from the first 5,000 stored ordinals. Every input
text and token-ID sequence is preserved. Batches retain input order and the
production limits of 32 inputs and 8,192 tokens. Context acceptance is 8,192
tokens, truncation is disabled, and no prompt is added. Every output is checked
for 1,024 finite, nonzero dimensions and L2 norm. Each arm warms the complete
sample before three timed repetitions, with arm order rotated across rounds.

The original Ollama model is the same Q8_0 GGUF and digest used by Phase C.
Alternatives use the official cached Hugging Face Qwen3-Embedding-0.6B weights,
revision `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418`, cast to float16. This
changes both precision and backend; the comparison does not isolate a pure
backend effect at identical arithmetic precision.

TEI 1.9.3 uses the checksum-verified official Homebrew Apple Silicon bottle,
native Metal, last-token pooling, two tokenizer workers and a persistent HTTP
client. It runs on a temporary loopback port. A local Sentence Transformers
runtime-length file sets the same 8,192-token acceptance limit; the cached
original model files are not modified. The [official TEI documentation](https://github.com/huggingface/text-embeddings-inference#apple-silicon-homebrew)
describes native Metal execution and token-based dynamic batching.

MLX 0.32.2 and its Metal runtime are installed only in an isolated experiment
directory. The probe copies the unchanged Qwen3 and base modules from
`mlx-embeddings` 0.1.0 into an isolated package, avoiding unrelated vision
loaders. It strictly loads all required model weights, uses right padding with
an attention mask, and applies last-token pooling and L2 normalization in
float32. Timings include tokenization, direct model computation, explicit GPU
evaluation, output transfer and validation. They exclude an MLX HTTP server,
model loading and warmup. This tests the model implementation with a small
native driver, not the complete upstream serving stack. See [MLX](https://github.com/ml-explore/mlx)
and the [embedding implementation](https://github.com/Blaizzy/mlx-embeddings).

## Measurements

| Probe | Configuration | Mean inputs/second |
| --- | --- | ---: |
| TEI comparison | Original Ollama Q8_0 | 16.4938 |
| TEI comparison | TEI float16, one input per request | 4.4464 |
| TEI comparison | TEI float16, production-sized batches | 3.0438 |
| MLX comparison | Original Ollama Q8_0 | 16.4693 |
| MLX comparison | MLX float16, one input at a time | 20.7173 |
| MLX comparison | MLX float16, production-sized batches | 25.0245 |

MLX batch repetitions were 25.0293, 25.0371 and 25.0070 inputs/second. TEI batch
repetitions were 3.0490, 3.0396 and 3.0429. TEI Metal is therefore not an
acceleration candidate for this workload/configuration. MLX's measured gain is
promising, but neither result is a full-corpus measurement or a universal
comparison of the frameworks.

During the TEI comparison, the system-wide AGX `Device Utilization %` counter
was sampled approximately every 0.5 seconds. Mean readings were 95.9% during
Ollama inference, 96.3% during TEI single-input inference and 99.4% during TEI
batch inference. These are coarse device activity readings, not per-process
attribution, achieved FLOPS or proof of maximum hardware efficiency. Higher
activity did not translate into higher throughput. Earlier GPU inactivity
occurred during CPU input preparation, before embedding computation.

## Numerical checks and deployment decision

All timed Ollama outputs matched the saved production vectors exactly in both
probes. MLX batch output differed from that cache by up to 0.00519584 in an
individual coordinate; mean cosine agreement was 0.9995920 and minimum was
0.9993597. MLX batch versus TEI single-input float16 reference had minimum
cosine agreement 0.9999902. MLX batch versus its own single-input reference had
minimum agreement 0.9999949. Detailed per-round differences and raw arrays
are archived. Close vector agreement does not establish equivalent rankings
or retrieval quality.

The current Phase C job remains on its original Ollama Q8_0 configuration.
The earlier accepted Phase C boundary freezes the embedding model/space and
retrieval legs. A switch to the float16 MLX path requires a separately specified
quality comparison, explicit backend/precision identity, an independent cache
and index, and matching query-time embedding. It must not silently mix new
vectors into the current cache or present changed-model results as the same
frozen baseline. No such switch or quality experiment was performed here.

If preserving the original implementation/vector configuration is required,
the previously tested two-instance Ollama path remains a smaller candidate:
16.8% higher sample throughput, with exact vector agreement on that sample.
That path also has not been enabled in the full job. Its evidence and the
current background preparation workflow are in the [execution note](phase-c-execution-note.md).

CPU preparation shared the host throughout these probes. The MLX driver
checked before and after timed rounds that the full job had not begun
embedding. No cloud resources were used. Temporary TEI ports were closed at
completion, the MLX process exited, and the original FiQA SQLite checksum
remained unchanged. These operational measurements are separate from product
regression counts and the Phase C fusion/validation results.

Evidence: [TEI measurements](../evaluation/phase-c/v1/tei-metal-audit.json),
[TEI artifacts](../evaluation/phase-c/v1/tei-metal-artifacts.json),
[GPU counter summary](../evaluation/phase-c/v1/tei-metal-gpu-summary.json),
[MLX measurements](../evaluation/phase-c/v1/mlx-audit.json) and
[MLX artifacts](../evaluation/phase-c/v1/mlx-artifacts.json). Raw vectors,
protocols, scripts, logs and checksums remain on the approved external disk.

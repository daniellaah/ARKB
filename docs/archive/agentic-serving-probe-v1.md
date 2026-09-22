# Serving concurrency and request repeatability probe v1

Register before execution. This is the first, bounded part of the development
concurrency study, following the completed v3 tool-readiness pilot. It measures
the current local model service under 1, 2 and 4 in-flight client requests. It
does not change Ollama's global settings, imply four actual GPU execution slots,
or estimate full Agent throughput from model requests alone.

Select the first pilot unit in schedule order for each of the four datasets,
retaining both MuSiQue variants. For each scenario retain the first submitted
request from F-S and A-All. This gives ten requests covering fixed evidence-heavy
generation and Agent tool selection. Selection uses schedule identity, never
success, answer quality, output length or latency. Copy actual requests unchanged
from verified provider journals; no answer labels or grading enter the probe.

Run three repetitions of the whole workload at each concurrency, 90 measured
requests total. The block order is 1/2/4, 2/4/1, 4/1/2. Within each repetition,
use one SHA-256 seeded ordering of the ten IDs shared by all concurrency levels.
Run one synthetic ready check before each block with the registered model and
context/options, recorded separately from the 90 measured requests. Warmups load
the model; prompt-cache behavior remains that of the current service and is not
assumed to represent a core run with unique questions.

Keep qwen3.5:4b, its identity, temperature zero, nonthinking mode, 32,768 context,
4,096 output ceiling, truncate=false, shift=false and the 330-second transport
timeout from v3. Do not tune settings after inspecting probe answers. Verify the
model identity and effective loaded context before/after each block. The runner
is frozen separately and imports the unchanged v3 source snapshot.

Measure completed requests per block wall second, client call latency, client
executor admission wait, raw provider durations/token counts, output-limit rate,
transport/protocol failures, normalized message equality across repeats, and
resident memory of local Ollama processes sampled during execution. Record model
VRAM allocation before/after each block. Provider total_duration and compute fields
are reported as supplied. Their residuals are not an exact server-queue measure;
server queue time is unavailable through this API. Do not label client admission
wait or unexplained duration as GPU queue time.

Acquire the original shared inference.lock for the complete probe. Check for
competing training/evaluation before each block and during it. Any transport,
provider-contract, identity, context, frozen-source or resource-competition failure
stops further dispatch after preserving in-flight results; pending work remains
unstarted, not scored as model failure. No automatic retry. Administrative stops
also finish in-flight calls. Output length, malformed generated finals or changed
tool choices are recorded model outcomes, not grounds for removing a request.

Report each repetition and each concurrency, with all outcomes retained. A level
can be considered for a subsequent full-Agent confirmation only if there are no
operational failures, and its median block throughput is at least 10% higher than
concurrency one without more length-limited outputs. This is a screening rule,
not authorization to change the core serving policy. Exact output stability does
not establish equal answer quality or justify one repetition for the core.

After this screen, register the complete Agent repeated-execution workload and
precision/effect scenarios. Use that evidence to choose question counts, repeated
trajectories and serving policy before freezing the replacement core protocol.

# Distillation pilot

Teach the 4B student to finish when it has evidence, by imitating trajectories
`deepseek-reasoner` produced on the real vault. A pilot: a hundred questions,
four samples each, enough to find out whether the pipeline is sound. It is not
sized to move a score, and the evaluation set's noise floor is about ±0.03
recall.

The evaluation corpus is the instrument and is never trained on. Questions
come from the Obsidian vault, which is a different corpus in a different set
of languages; what is supposed to transfer is retrieval behaviour, not the
notes.

```
training/
  chat_format.py     one rendering of a conversation, shared by serving and training
  transport.py       the student's endpoint, and a sampled teacher
  make_questions.py  1  a question per vault note, filtered by retrieval
  rollout.py         2  parallel trajectories from the teacher, scored and filtered
  build_dataset.py   3  trajectories -> tokens + loss mask
  train_lora.py      4  LoRA on the student
  eval_sft.py        4  the unchanged evaluation, against the served student
  data/ models/      produced artifacts, gitignored
```

## Environment

Training runs in its own virtual environment, so the project's `uv.lock` stays
the truth about what the product needs:

```bash
uv venv --python 3.12 .venv-train && VIRTUAL_ENV=.venv-train uv pip install mlx-lm
```

`make_questions`, `rollout` and `eval_sft` run in the project's `.venv`,
because they drive the product. `build_dataset` and `train_lora` run in
`.venv-train`, because they need the tokenizer and MLX.

## The pipeline

```bash
# 1. questions from the vault (local 27B, no paid API)
.venv/bin/python -m training.make_questions --notes 320 --keep 100

# 2. trajectories from the teacher, eight at a time, under a hard cost ceiling
.venv/bin/python -m training.rollout --samples 4 --workers 8 --max-cost 3.0

# 3. samples, one per model request, with the loss mask computed and checked
.venv-train/bin/python -m training.build_dataset training/data/rollouts-pilot \
    --out training/data/sft-pilot --max-tokens 16384 --show 2

# 4. LoRA, then the evaluation against the served student
.venv-train/bin/python -m training.train_lora --data training/data/sft-pilot \
    --adapter training/models/sft-pilot-4b --max-seq-length 16384
.venv-train/bin/python -m mlx_lm server --model mlx-community/Qwen3.5-4B-bf16 \
    --adapter-path training/models/sft-pilot-4b --port 8080 --max-tokens 4096 &
.venv/bin/python -m training.eval_sft --label sft-pilot-4b

# the control: the same untrained weights through the same endpoint, so the
# comparison measures the adapter and not bf16 against Ollama's Q4_K_M
.venv-train/bin/python -m mlx_lm server --model mlx-community/Qwen3.5-4B-bf16 \
    --port 8080 --max-tokens 4096 &
.venv/bin/python -m training.eval_sft --label sft-base-4b-mlx
.venv/bin/python -m evaluation.eval compare evaluation/results/sft-base-4b-mlx \
    evaluation/results/sft-pilot-4b
```

## What the serving path costs

The student is served by `mlx_lm.server`, an OpenAI-compatible endpoint, and
reached with the product's own `ChatCompletionsClient` pointed at it. Four
adaptations were needed, all in `chat_format.py` and all applied identically
when rendering training samples, which is the point: train and serve cannot
drift if they call the same function.

1. Qwen3.5's template raises on a system message that is not the first one,
   and the agent loop appends system messages mid-conversation. They become
   user messages.
2. Tool arguments travel as JSON strings; the server parses them back into
   dicts before templating.
3. A parameter typed `["string", "null"]` is advertised as `"string"`. Tool
   calls arrive as XML with no types, and the server's parser recovers them
   from the advertised schema; a union type falls through to
   `ast.literal_eval`, which rejects a source path and silently drops the call.
4. The reserved finalization asks for thinking off through
   `chat_template_kwargs`, which is how this template takes what Ollama takes
   as `think=False`.

## What the student's architecture costs

Qwen3.5 is a hybrid: twenty-four of its thirty-two blocks run a gated delta
rule whose MLX kernel has no backward pass, so `mlx_lm` substitutes a
sequential reference implementation for any block in training mode. A LoRA
below the last block must therefore differentiate through that
implementation. Measured on an M4 Max, one sample, forward and backward:

| LoRA depth | 2k tokens | 4k | 8k |
| --- | --- | --- | --- |
| last block only | 13.4 GB, 1.6 s | 18.5 GB, 3.4 s | 29.6 GB, 7.3 s |
| last four blocks, checkpointed | 39.9 GB, 7.2 s | 100 GB, 21 s | 310 GB, 417 s |

PyTorch on MPS has a differentiable chunked implementation and is correct, but
took 19.8 s for a 2k sample, which is 25 hours for this pilot's schedule.

So the pilot trains the last block only, and `train_lora.py` keeps every
frozen block in evaluation mode so it stays on the fast kernel. Anything
larger than this pilot needs a backward pass for the gated delta rule, or a
student that is not hybrid.

## What the pilot measured

100 questions from the vault, four samples each, `deepseek-reasoner` at
temperature 0.7: 392 trajectories in 12.3 minutes on 8 workers for $4.45, of
which 358 were kept (91.3%) by `source_recall == 1.0` and a status of
`answered` or `partial`. Those became 1,435 samples (1,307 train, 128
validation, split by question), 11.8M tokens of which 488,542 carry loss.
Not one sample failed the boundary check. LoRA on the last block, rank 32,
one epoch, 326 optimizer updates, 2.4 hours: validation loss 0.756 -> 0.678.

The comparison has three arms, because the obvious two would have been
misread. `sft-base-4b` is the 4B baseline through Ollama at Q4_K_M;
`sft-base-4b-mlx` is the *same untrained weights* at bf16 through the serving
path this pilot uses; `sft-pilot-4b` is the adapter on top of that.

| | sft-base-4b (Ollama Q4) | sft-base-4b-mlx (bf16) | sft-pilot-4b (+LoRA) |
| --- | ---: | ---: | ---: |
| answered | 0.763 | 0.877 | 0.868 |
| source recall | 0.833 | 0.918 | 0.936 |
| source precision | 0.960 | 0.954 | 0.959 |
| tool calls | 5.17 | 4.68 | 4.69 |
| errors | 3 | 5 | 3 |
| s/question | 16.5 | 26.9 | 27.4 |

Read against the Ollama baseline the adapter looks like a success: +0.105
answered, +0.103 recall, -0.47 tool calls. Read against the control it is
nothing: -0.009 answered (3 wins, 107 ties, 4 losses), +0.018 recall (8 wins,
7 losses), +0.018 tool calls. Almost the whole apparent gain is bf16 rather
than Q4_K_M, and it would have been credited to the fine-tune by any
comparison that did not hold the serving path fixed.

The behaviour did not move either. Nineteen questions finished in fewer tool
calls without losing recall; twenty-five took more without gaining any.
Rejected `finish` proposals, the signature of the citation contract being
broken, were 65 before and 66 after. What did not appear is as informative:
fabricated tool output stayed at one occurrence, repeated identical calls at
21, truncated responses at one, and error finals fell from five to three. The
pilot did not teach the model to invent its environment.

Individual trajectories do show the intended shape. On `pilot_042` the
control searched, read, searched twice more, read again and answered with
half the expected notes in seven tool calls; the adapter searched, read the
policy note and finished, with both notes, in three. On `pilot_033` the
control proposed the same `finish` five times, was refused five times for
citing nothing, and ended in error; the adapter read once and finished. The
regressions are equally real: on `pilot_019` the adapter wandered into two
extra `list` calls and cited a reference that did not exist.

That is what 358 trajectories on one transformer block buys, and it is the
expected result: the evaluation's noise floor is about 0.03 recall, and the
architecture limits training to 1.9M parameters in the last of 32 blocks.
Before this is worth scaling to a thousand questions, the gated delta rule
needs a backward pass so a LoRA can reach the whole network.

## What the second round measured

The first round could not separate "the fine-tune does nothing" from "the
adapter had no capacity" or "the corpus had no headroom". The second round
removes all three excuses and the answer does not change.

What changed between the rounds: the agent's retrieval and citation defaults
were fixed first (see the repository history), so the teacher demonstrates a
policy worth imitating; the teacher's trajectories were regenerated against the
fixed tools (400 trajectories, 88.7% kept, $3.44, 649.8 s at 8 workers, 23%
cheaper per trajectory than the first round because a fixed default shortens
them); and the student became **Qwen3-4B**, which is full attention, so LoRA
reaches all 36 blocks — 66M trainable parameters against the first round's 1.9M
in a single block.

The dataset's turn distribution had to be chosen deliberately. At a 8,192-token
cap, 8.5% of samples came from turn 4 or later and **not one finalization turn
survived**: training on that would teach a model that already stops too early
to stop even earlier. At 16,384 tokens with two trajectories per question the
distribution is usable — 708 samples spanning turns 1 to 8, 47 of them
finalizations, 257k tokens carrying loss, no boundary failures.

Training: all 36 blocks, rank 32, scale 2.0, gradient checkpointing, one epoch,
708 iterations in 7.5 hours at 91 GB peak. **Validation loss 1.606 → 0.794**, a
51% reduction against the first round's 10%. The adapter learned the teacher's
token distribution on questions it never saw.

The task behaviour did not follow.

| | 180 vault questions | | 114 dev questions | |
| --- | ---: | ---: | ---: | ---: |
| | base | +LoRA | base | +LoRA |
| delivered recall | 0.589 | 0.556 | 0.782 | 0.825 |
| cited recall | 0.544 | 0.506 | 0.725 | 0.760 |
| answered | 0.756 | 0.761 | 0.763 | 0.772 |
| tool calls | 1.55 | 1.58 | 1.98 | 1.97 |
| runs calling no tool | 44 | 43 | 10 | 8 |

The two corpora move in opposite directions — the vault by −0.039 cited recall,
the development set by +0.035 — which is what noise looks like. Pooled over all
**294 paired questions: 26 wins, 234 ties, 26 losses** on cited recall; 23/21 on
delivered recall; 21/19 on answered. The behaviour the training set was selected
for is untouched: 54 runs called no tool before, 51 after, and 38 of them are
the same questions.

### Where the signal went

The training set says the right thing. All 175 turn-1 samples call a tool;
the teacher never answers from memory. But turn 1 is one short decision and the
rest of a trajectory is long prose:

| | turn 1 | turns 3-5 |
| --- | ---: | ---: |
| share of samples (the loss is a per-sample token mean, so this is the gradient's share) | 24.7% | 46.2% |
| share of tokens carrying loss | 8.5% | 64.3% |
| mean tokens carrying loss per sample | 113 | ~400 |

The failure that matters is a single early branch — search, or answer from what
the model already knows. Imitation spends its capacity on the middle of
trajectories the student never reaches, because it stops after 1.55 tool calls.
A 51% drop in validation loss is consistent with learning the reasoning and
tool-call *style* of turns 3 to 5 while leaving the turn-1 decision where it was.

So the honest conclusion is narrower than "SFT does not work here" and more
useful: **token-weighted imitation of positive trajectories is the wrong
instrument for changing one early decision.** What would address it, in order of
how much machinery each needs: weight the loss by decision rather than by
sample; train only on the turns where the student and the teacher diverge; or
give the objective a contrast — the rollouts already contain 45 rejected
trajectories per round, so the same pipeline produces preference pairs on the
question where the student went wrong.

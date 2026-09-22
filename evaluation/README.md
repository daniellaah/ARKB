# Evaluation

One question file, one corpus, one script. It answers one question: did a
change to the agent (tools, prompt, model, budget) make it better or worse on
the same 84 questions?

```
evaluation/
  notes/           58 Markdown notes, the corpus (40 authored notes, 18 fixtures)
  questions.json   84 questions with the notes they should cite
  eval.py          run, rescore, compare
  test_eval.py
  results/         run outputs (gitignored)
```

## Questions

```json
{"id": "pilot_001", "type": "semantic_discovery",
 "question": "I want the assistant to decide its next step from information it just obtained ...",
 "question_zh": "我想让助手根据刚获得的信息决定下一步 ...",
 "expected_sources": ["01_workflows_and_agents.md"],
 "reference": "模型随新信息选择下一步动作，预定义步骤属于 workflow。"}
```

| type | n | expects |
| --- | ---: | --- |
| semantic_discovery, exploratory_retrieval, knowledge_qa, multi_hop_qa | 12, 12, 12, 6 | an answer citing the expected notes |
| exact_lookup | 30 | exactly the notes containing the literal term (24 of them generated from corpus term frequencies, truth by substring) |
| direct_read | 4 | read the named note only, no search |
| evidence_gap | 4 | `insufficient_evidence` when nothing is expected; `partial` when part of the question is answerable |
| no_retrieval | 4 | an answer without any retrieval tool |

`expected_sources` is read only by the scorer, after a run. `reference` is
for a human reading the results; it is not scored. Every question was written
during development and the labels have not been independently reviewed:
this set catches regressions and large effects, it does not certify quality.
Add questions by editing the file; `test_eval.py` checks that every expected
note exists.

## Commands

```bash
.venv/bin/python -m evaluation.eval run --label my-change --model qwen3.5:9b --think
```

Builds (or reuses) the index of `notes/`, runs the product agent on every
question (temperature 0, 32,768 context, 4,096 output tokens, 8 turns,
budgets 12/10/6 tool, query and read calls, 8,000 evidence tokens, 300 s) and
writes `results/<label>/{run.json,results.jsonl,summary.json,summary.md}`.
About ten minutes at 9B. `--limit N` takes N questions per type for a quick
check; `--resume` continues an interrupted run; `--max-evidence-tokens`,
`--max-turns`, `--num-ctx`, `--num-predict` override the defaults. A finished
run is not rerun under the same label.

```bash
.venv/bin/python -m evaluation.eval compare evaluation/results/baseline evaluation/results/my-change
```

Per-question differences (B minus A) on the shared questions: means, the
difference and wins/ties/losses, overall and per type.

```bash
.venv/bin/python -m evaluation.eval rescore evaluation/results/my-change
```

Recomputes the scores of a saved run after the scorer or the question file changed.

## Metrics

| metric | meaning |
| --- | --- |
| answered | final status `answered` or `partial` |
| source_recall / source_precision | expected notes among the cited notes, and cited notes that were expected |
| delivered_recall | expected notes among the notes whose evidence reached the model (what retrieval found, before the answer) |
| complete | exact_lookup: cited set equals the expected set |
| gap_respected, no_retrieval, read_only | behaviour on evidence_gap, no_retrieval and direct_read questions |
| elapsed_s, model_requests, tool_calls, prompt/eval tokens, evidence_tokens, responses_cut | cost; `responses_cut` counts model responses stopped at `num_predict` |

An error (harness exception or error final) counts as not answered with zero
recall. `results.jsonl` keeps the full trace of every question (requests,
tool calls, delivered evidence, the final object), so any number can be
traced back to what the agent did.

## Reading the numbers

Single runs are noisy: an earlier repeatability study found two thirds of
questions change their final object between identical runs of a small
model. Trust differences that are large and consistent across types; treat
a few wins against a few losses as noise. For anything that needs intervals
or a held-out set, start from git history (`docs/archive.md`).

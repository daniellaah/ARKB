# Development evaluation

One purpose: tell whether a change to the agent (tools, prompts, model,
budgets) made it better or worse, in minutes, with paired comparisons. It
evaluates the product agent as it ships; an ablation is a product
configuration, not an evaluation adapter. Design and rationale:
[docs/eval-design.md](../docs/eval-design.md).

## Devset

| Slice | n | Inputs | Primary metric | Core |
| --- | ---: | --- | --- | --- |
| v2 | 60 | in repo (`devset/v2-pilot`, 58 notes) | span evidence coverage (delivered); cited source recall; behaviour by task | yes |
| exact-v2 | 24 | in repo (terms from the v2 corpus; substring truth) | completeness of cited sources | yes |
| exact-nfcorpus | 24 | `/Volumes/ARKBPhaseC` (3,633 documents) | completeness of cited sources | optional |
| recall-nfcorpus, recall-fiqa | 20 + 20 | volume | positive-document recall (delivered) | optional |
| musique | 20 | volume (prepared contexts) | first-line answer F1; answerability; support F1 | optional |
| long-browsecomp | 10 | volume | positive-document recall (delivered) | optional |

Every ID is exposed development material chosen by hash, never by outcome;
the v2 labels are provisional (assistant-authored, unreviewed) until the
review pass marks them. `devset/scenarios.json`, `labels.json`, `scopes.json`
and `manifest.json` are committed; the scoring source maps under
`devset/scoring/` are derived (13 MB) and rebuilt by the build command.
Labels are read by `score.py` after inference and by nothing else.

## Commands

```bash
.venv/bin/python -m evaluation.devset.build --index
```

Rebuilds the devset and the v2 corpus index. Without the volume the optional
slices are carried over from the committed devset unchanged.

```bash
.venv/bin/python -m evaluation.run run --label <label> --model qwen3.5:9b --think
```

Runs the product agent (temperature 0, 32,768 context, 4,096 output tokens;
budgets 12/10/6 tool, query and read calls, 8,000 evidence tokens, 300 s;
8 turns) and writes `results/<label>/{run.json,results.jsonl,summary.json,summary.md}`.
`--slices core` (default) is v2 + exact-v2, about ten minutes at 9B;
`--slices all` adds every optional slice whose inputs are present and lists
the skipped ones in `run.json`; a comma-separated list selects slices
explicitly. `--limit N` takes N scenarios per slice for a quick check;
`--resume` continues an interrupted run; the budget flags override the
defaults. Completed runs are immutable: rerun under a new label.

```bash
.venv/bin/python -m evaluation.run rescore results/<label>
```

Recomputes scores and summary from the saved records with the current scorer.

```bash
.venv/bin/python -m evaluation.compare --run base=results/<a> --run change=results/<b> --output comparisons/<name>.md
```

Per-scenario paired differences with slice-stratified percentile bootstrap
intervals (seed 20260912, 20,000 resamples, nominal, uncorrected) and
wins/ties/losses, per slice and metric, plus pooled cost differences; writes
Markdown and JSON. Any number of runs; `--contrast LEFT:RIGHT` picks the
contrasts (default: every later run minus the first).

```bash
.venv/bin/python -m evaluation.review export --run a=results/<a> --run b=results/<b> --salt <name> --output devset/review/<name>
.venv/bin/python -m evaluation.review import devset/review/<name>/sheet.md --mapping devset/review/<name>/mapping.json --output devset/review/<name>-verdicts.json --reviewer <you>
.venv/bin/python -m evaluation.review agreement devset/review/<name>-verdicts.json --run a --results results/<a>/results.jsonl
```

Blinded human review: a hash-chosen sample (default 20 v2 + 10 MuSiQue), the
runs' answers shuffled per question, the mapping in a separate file. Mark one
box each for *correct* and *grounded* in the sheet, import it, and read the
agreement between a run's automatic outcome and the human verdicts.

`engine_recall.py` measures the retrieval engine alone (no model) on the
recall scenarios at 5, 10 and 20 results per mode.

## Reading the numbers

- Delivered coverage and recall count evidence that entered the conversation;
  cited, answered and answer metrics score the final object. An error final
  scores 0 on the latter; `errors` in a summary counts error finals, and
  `responses_cut` counts model responses stopped at `num_predict`.
- Single runs are noisy (the repeatability study measured two thirds of cells
  changing their final object between identical runs): read the intervals
  from `compare`, not single wins.
- Nothing here is a quality claim; use it for before/after checks,
  regressions and cost drift.

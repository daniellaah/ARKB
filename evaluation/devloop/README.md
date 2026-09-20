# Fast development loop

A development instrument, not a study: about 150 scenarios, minutes per run, no
LLM judge, no statistical claims. It measures **evidence delivery**,
**completeness** and **cost** of the current product agent so that tool and
retrieval fixes can be checked in one sitting. Labels live in `labels.json` on
the scoring side; the runner reads them only after inference.

## Slices (devset-v1)

| Slice | Scenarios | Scope | Primary metric | Labels |
| --- | ---: | --- | --- | --- |
| v2 | 60 | 58-note v2 pilot corpus, English-translated queries | span evidence coverage (delivered) | provisional v2 spans and qrels |
| exact-nfcorpus | 24 | NFCorpus, 3,633 documents | completeness of cited sources | literal substring truth computed from the corpus |
| recall-nfcorpus | 20 | NFCorpus full index | positive-document recall (delivered) | public qrels, pilot and reserve IDs |
| recall-fiqa | 20 | FiQA full index | positive-document recall (delivered) | public qrels, pilot and reserve IDs |
| musique | 20 | ten MuSiQue pairs, prepared contexts | answer F1, answerability, pair sufficiency | official-style labels |

Every ID here is exposed development material. Selection uses IDs, hash order
and corpus text statistics only. The v2 translations are assistant-authored and
unreviewed; Chinese originals are kept beside them.

## Commands

```bash
.venv/bin/python -m evaluation.devloop.build --index
```

```bash
.venv/bin/python -m evaluation.devloop.run run --output evaluation/results/devloop/<label> --label <label>
```

```bash
.venv/bin/python -m evaluation.devloop.run compare evaluation/results/devloop/<a> evaluation/results/devloop/<b> --output evaluation/devloop/comparisons/<a>-vs-<b>.md
```

```bash
.venv/bin/python -m evaluation.devloop.bootstrap --run A-All=evaluation/results/devloop/<a> --run F-H=evaluation/results/devloop/<b> --contrast A-All:F-H --output evaluation/devloop/comparisons/<name>.md
```

`--adapter product` (default) runs the product agent loop with all tools, the
product system prompt, temperature 0, 32,768 context and 4,096 output tokens,
under the same 12/10/6 tool, 8,000-token evidence and 300-second budgets as the
registered studies. `--adapter contract:A-All` (or any arm ID) runs the
evaluation adapter of that arm instead, with the same model, thinking and
options as the command line asks for (the registered runner keeps its own
frozen constants). `--model` and `--think` change the model; `--slices`
and `--limit` select a subset for quick checks; `rescore` recomputes the
scores of a saved run with the current scorer. `bootstrap` pairs any number of
runs per scenario and reports slice-stratified percentile intervals (nominal,
uncorrected); `review_sheet` exports a blinded human-review sheet with a
hash-chosen sample and a separate arm mapping. The `devset-v1/scoring/` source maps
are derived files (13 MB) that the build command regenerates; they are not
committed. Runs record the git head, dirty
state, source hashes and model identity. Results are gitignored under
`evaluation/results/devloop/`; keep comparison Markdown under
`evaluation/devloop/comparisons/`.

Use it for: before/after checks of a fix, regressions, cost drift. Do not use it
for: claims about answer correctness, population estimates, or selecting among
arms by quality.

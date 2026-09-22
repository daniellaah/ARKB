# Development evaluation: design (v2)

Decided 2026-09-21. One evaluation system, built for one purpose: tell
whether a change to the agent (tools, prompts, model, budgets) made it better
or worse, in minutes, with paired comparisons. It replaces the registered-study
library, the evaluation package inside the product and the historical
experiment trees, which together hold about 21,000 lines of which the
development loop uses under 2,000.

## Non-goals

No registration, readiness gates, judge models, stop controllers, power
analysis or arm adapters. No quality claims: the scenarios are exposed
development material and the v2 labels are provisional until reviewed. A
study that needs any of these starts from this system and adds them outside it.

## Layout

```
evaluation/
  README.md          usage and conventions, one page
  __init__.py
  common.py          paths, strict JSON, digests, the ID hash, the run deadline
  v2.py              the v2 pilot dataset loader and span-coverage scorer
  devset/
    build.py         builds scenarios.json, labels.json, scopes.json, manifest.json
    scenarios.json   one record per question (id, slice, scope, query, task fields)
    labels.json      id -> labels; read only by score.py, never by run.py
    scopes.json      scope -> corpus, sqlite, vault_id, optional index manifest
    manifest.json    slice counts, source digests, build identity
    v2-pilot/        the 58-note corpus, queries, evidence spans, qrels (moved from evaluation/data/v2/pilot)
    review/          human verdicts (blinded sheet + filled verdicts), see below
  run.py             runs the product agent on a devset; results.jsonl + run.json
  score.py           per-scenario metrics from a saved record; summary per slice
  compare.py         paired differences, slice-stratified bootstrap, wins/ties/losses
  transport.py       Ollama chat client with explicit model/options/think; model identity
  review.py          blinded review-sheet export, verdict import, agreement
  engine_recall.py   engine-only recall at 5/10/20 on the recall scenarios (no model)
  tests/             runner, scorer, comparison, build and review checks (no services)
  comparisons/       committed comparison Markdown and JSON
  results/           gitignored run outputs
```

Target size: about 1,000 lines plus tests. Everything the loop needs from the
old packages moves in with it: `write_json`/`digest`, the v2 span scorer
(`load_dataset`, `evidence_scores`), the MuSiQue token F1, the run deadline,
the delivered-evidence, cost and MuSiQue scorers, `model_identity`, the ID
hash. Each is copied without its registered-study coupling (no `ARM_BY_ID`,
no seven-arm schedule); the first-positive-discovery diagnostic is dropped.

## Data

Scenario record (unchanged shape, one format for every slice):

```json
{"id": "pilot_001", "slice": "v2", "scope": "v2-pilot", "task_type": "semantic_discovery",
 "query": "...", "query_zh": "...", "optional": false}
```

Slices, with what they measure and where their inputs live:

| Slice | n | Inputs | Primary metric | Optional |
| --- | ---: | --- | --- | --- |
| v2 | 60 | in repo (`devset/v2-pilot`) | span evidence coverage, delivered; cited source recall; behaviour | no |
| exact-v2 | 24 | in repo (terms drawn from the v2 corpus; truth is substring truth) | completeness of cited sources | no |
| exact-nfcorpus | 24 | `/Volumes/ARKBPhaseC` (3,633 documents; the hard version) | completeness of cited sources | yes |
| recall-nfcorpus | 20 | volume | positive-document recall, delivered | yes |
| recall-fiqa | 20 | volume | positive-document recall, delivered | yes |
| musique | 20 | volume (prepared contexts) | answerability; first-line answer F1; support F1 | yes |
| long-browsecomp | 10 | volume | positive-document recall, delivered | yes |

The core set (v2 + exact-v2, 84 questions) runs anywhere the repository is
checked out and takes about ten minutes at 9B. Optional slices run when the
volume is mounted and are skipped with a note in `run.json` otherwise; their
scoring source maps (13 MB) stay derived and gitignored. The devset is built
once by `build.py` from the in-repo v2 files and, when present, the frozen
pilot-v3 selection on the volume; `scenarios.json`, `labels.json`,
`scopes.json` and `manifest.json` are committed, so a run never needs the
build inputs. Without the volume, `build.py` rebuilds the core slices and
carries the optional slices over from the committed devset unchanged, so a
laptop never loses them; selection hashes keep the identities of devset-v1,
so every scenario ID of the earlier runs is preserved.

Labels never enter `run.py`. `score.py` reads them after inference. Selection
of scenarios uses IDs, hashes and corpus statistics only.

## Metrics

Kept because they carried information in the development studies so far:

| Slice | Metric | Definition |
| --- | --- | --- |
| v2 | `evidence_coverage_delivered` | fraction of required evidence spans covered by evidence delivered to the conversation |
| v2 | `source_recall_cited` | fraction of expected sources among the final's citations |
| v2 | `answered`, `abstained`, `no_retrieval_respected`, `read_only_respected` | behaviour by task type |
| exact-* | `completeness_cited`, `spurious_cited`, `complete_and_exact` | cited sources against substring truth |
| recall-*, long | `positive_recall_delivered` | positive documents among delivered sources |
| musique | `answerability_correct`, `answer_f1_first_line`, `answer_em_first_line`, `support_f1` | canonical final against gold |
| all | `elapsed_s`, `model_requests`, `tool_calls`, `prompt_tokens`, `eval_tokens`, `delivered_evidence_tokens`, `final_status` | cost and outcome |

Dropped: the returned/submitted stages (delivered is what the model saw;
cited is what it used), per-stratum breakdowns (available from the records
when needed), full-text answer F1 (the first-line contract replaces it),
completed retrieval legs, mode transitions, judge scores.

An error final scores 0 on cited/answer metrics and "not answered" on
behaviour; delivered metrics count whatever entered the conversation before
the failure. A summary's `errors` counts error finals (the old loop counted
only harness exceptions), and responses cut at `num_predict` are counted
separately.

## Runner

`run.py` evaluates the product agent only: `Runtime.run_agent` with the
product tools, prompt and loop. Its configuration surface is what the product
exposes: `--model`, `--think`, `--num-ctx`, `--num-predict`, `--max-turns`,
the five budgets (tool, query, read calls; evidence tokens; seconds), and,
once the product grows the option, `--tools` to restrict the tool set. An
ablation is a product configuration, not an evaluation adapter.

Every run writes `run.json` (label, configuration, git head and dirty files,
source digest of `src/` and `evaluation/`, model identity from Ollama, devset
manifest, wall time, skipped optional slices) and `results.jsonl` (one record
per scenario with the full observation trace, the canonical final and the
scores). Completed runs are immutable; `--resume` continues an interrupted run
in the same directory; `rescore` recomputes scores from saved records with the
current scorer and records the scorer digest.

`transport.py` sends exactly what the loop asks (the caller's `think` flag
wins), adds `num_ctx`/`num_predict`, and refuses a request for another model.

## Comparison

`compare.py` pairs two or more runs per scenario, reports per slice and
metric the mean difference, a slice-stratified percentile bootstrap interval
(seed 20260912, 20,000 resamples, nominal, uncorrected) and wins/ties/losses,
and pooled cost differences stratified by slice. Output: one Markdown table
and its JSON, committed under `comparisons/`.

## Human review

`review.py` exports a blinded sheet (questions chosen by ID hash, answers of
several runs shuffled per question, arm mapping kept in a separate file) and
imports filled verdicts (`correct`, `grounded`, notes per answer) into
`devset/review/<name>-verdicts.json`, unblinded per run. The 30-question
sheet already exported for the agentic-versus-workflow runs (20 v2, 10
MuSiQue) is the first calibration pass: its verdicts become the first
reviewed subset of the v2 labels, and `review.py agreement` reports, for a
reviewed run, how often its automatic `answered`/`abstained` outcome matched
the human `correct` verdict. The review itself is done by a person; nothing
here substitutes for it.

## Migration

Four commits, each leaving `make test` and `make lint` green:

1. **Move the loop.** Create the new modules from `evaluation/devloop` plus
   the functions listed above; add the exact-v2 slice to the build; rebuild
   the devset and check that `rescore` of an existing run reproduces its
   summary; point `Makefile`, `README.md` and the project map at the new
   entry points.
2. **Delete.** `src/arkb/evaluation`, `tests/evaluation`,
   `evaluation/agentic_tools`, `evaluation/studies`, `evaluation/experiments`,
   `evaluation/audits`, `evaluation/devloop`, `evaluation/data/v2/core-intake`
   and the `agent_v1.jsonl` fixture. Registered-run artifacts under
   `evaluation/agentic-tools/` (3.5 MB of protocols, accounting and audits
   that the archived reports cite) move to `archive/agentic-tools/`.
3. **Archive the documents.** Historical reports and protocols move to
   `docs/archive/`; `docs/archive.md` indexes them with the commit that last
   ran their code. `docs/` keeps the project map, the system review, the
   fix log, the agentic-versus-workflow report and this design.
4. **Calibrate.** Import the filled 30-question verdicts, mark the reviewed
   v2 cases, record the agreement figure in the README.

Frozen study products on `/Volumes/ARKBPhaseC` are not touched; the git
history keeps every deleted file.

## Risks

- Registered studies can no longer be rerun from the working tree. Their
  source snapshots on the volume and the git history cover reproduction.
- The core set is small (84 questions) and single-run noise is real (the
  repeatability study measured two thirds of cells changing their final
  object between identical runs); read intervals, not single wins.
- Optional slices depend on an external volume; a laptop without it measures
  only evidence coverage and enumeration, not document recall or multi-hop.

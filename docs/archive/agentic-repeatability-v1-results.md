# Full Agent repeatability v1 results

The registered 70 new trajectories finished on 2026-09-14 at 08:43:51 UTC.
The worker exited successfully. Together with 35 pinned v3 trajectories, this
provides three executions of each of 35 scenario/arm cells. These are only four
independent questions: one per dataset, with both MuSiQue context variants.

The independent [completion audit](../evaluation/agentic-tools/repeatability-v1/completion-audit.json)
verified all 105 attempt identities and result/provider checksums, reconstructed
397 actual HTTP request/response pairs, checked the effective model/context,
replayed 1,169 trace checks and independently reconstructed the normalized
comparison fingerprints. All 31 scope preparations retained their own auxiliary
check artifact: 310 fixture calls passed. The auxiliary archive naming repair
worked; earlier v3 artifacts remain unchanged.

The new trajectories made 234 executed tool calls: 227 valid calls and seven
model input errors (six unavailable source names and one invalid reference).
There were no operational tool errors, valid-call deadline failures or final
execution failures. This finite workload establishes a passed engineering gate,
not a population reliability guarantee.

## Repeated execution differences

Known opaque evidence references were mapped to full source/content/revision/span
identities only during offline comparison. Model inputs and outputs were not
rewritten. Unknown references remain visible.

| Comparison | All three executions identical / 35 | Two new executions identical / 35 |
|---|---:|---:|
| Complete final object | 10 | 11 |
| Executed tool sequence, including `finish` arguments | 15 | 15 |
| Evidence source coverage | 30 | 30 |
| Returned/delivered/submitted evidence identities | 27 | 28 |
| Model request count | 30 | 30 |
| Trajectory stop reason | 34 | 35 |
| Final status | 31 | 33 |

The registered tool fingerprint includes `finish`, so changed answer text can
change that fingerprint without changing retrieval decisions. A supplementary
post-run calculation excludes `finish`: 25/35 cells have identical search/read
actions in the two new runs. Among the 25 Agent cells, 15/25 are identical; all ten
fixed-retrieval cells have identical retrieval actions. This diagnostic does not
replace the registered comparisons.

Exact output differences are not correctness differences. Different sessions
receive fresh opaque reference IDs and provider tool-call IDs; later requests
can therefore differ even when their bound evidence has the same meaning. This
study measures whole-session repeatability under the actual product contract,
not generation randomness under byte-identical model inputs. Serving-v1's stable
replayed messages do not establish whole-Agent determinism.

## Cost and interpretation

Measured new trajectory time was 694.33 seconds (11.57 minutes); scope setup was
409.88 seconds (6.83 minutes). Total worker wall time was 1,214.10 seconds
(20.24 minutes), including remaining fixtures and verification. Setup reused
the full existing corpora/indexes; no embeddings were rebuilt. Differences from
v3 latency are not a randomized speed comparison because caches and run times
changed. The selected four questions are not a population cost estimator.

The findings rule out assuming deterministic whole-Agent behavior. They do not
estimate population repeated-run correlation or establish answer quality. Core
sample and repetition counts must follow a separately stated estimand and
precision/cost calculation. No core dispatch is authorized by this study alone.

Raw records: `/Volumes/ARKBPhaseC/agentic-tools-v1/repeatability-v1`.
Protocol SHA-256: `16cc2c1ffcdf5819eaec0d304b770fa8e2313a47f613c4f3a9f73b96dd8b6bcc`.

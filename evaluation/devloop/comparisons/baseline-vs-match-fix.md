# Dev loop comparison: `baseline-4b-nothink` vs `match-fix-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=befa4d457c+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.722 | 1.000 | +0.278 | 13 | 11 | 0 |
| exact-nfcorpus | completeness_returned | 0.722 | 1.000 | +0.278 | 13 | 11 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 8.4 | 17.7 | +9.4 | | | |
| musique | answer_f1 | 0.073 | 0.072 | -0.001 | 4 | 3 | 3 |
| musique | answerability_correct | 0.550 | 0.550 | +0.000 | 0 | 20 | 0 |
| musique | support_f1 | 0.267 | 0.270 | +0.003 | 3 | 5 | 2 |
| musique | elapsed_seconds_mean | 11.1 | 10.9 | -0.2 | | | |
| recall-fiqa | positive_recall_delivered | 0.311 | 0.294 | -0.017 | 0 | 19 | 1 |
| recall-fiqa | positive_recall_returned | 0.311 | 0.294 | -0.017 | 0 | 19 | 1 |
| recall-fiqa | elapsed_seconds_mean | 10.6 | 11.1 | +0.5 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.147 | 0.152 | +0.005 | 2 | 17 | 1 |
| recall-nfcorpus | positive_recall_returned | 0.147 | 0.152 | +0.005 | 2 | 17 | 1 |
| recall-nfcorpus | elapsed_seconds_mean | 10.6 | 11.7 | +1.1 | | | |
| v2 | evidence_coverage_delivered | 0.926 | 0.947 | +0.021 | 2 | 44 | 1 |
| v2 | evidence_coverage_returned | 0.926 | 0.947 | +0.021 | 2 | 44 | 1 |
| v2 | source_recall_cited | 0.933 | 0.846 | -0.087 | 2 | 43 | 7 |
| v2 | elapsed_seconds_mean | 6.1 | 6.7 | +0.7 | | | |

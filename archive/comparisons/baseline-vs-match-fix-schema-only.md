# Dev loop comparison: `baseline-4b-nothink` vs `match-fix-schema-only-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=befa4d457c+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.722 | 0.966 | +0.243 | 12 | 12 | 0 |
| exact-nfcorpus | completeness_returned | 0.722 | 0.966 | +0.243 | 12 | 12 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 8.4 | 16.8 | +8.4 | | | |
| musique | answer_f1 | 0.073 | 0.065 | -0.008 | 3 | 4 | 3 |
| musique | answerability_correct | 0.550 | 0.550 | +0.000 | 0 | 20 | 0 |
| musique | support_f1 | 0.267 | 0.180 | -0.087 | 0 | 8 | 2 |
| musique | elapsed_seconds_mean | 11.1 | 10.1 | -1.0 | | | |
| recall-fiqa | positive_recall_delivered | 0.311 | 0.291 | -0.021 | 0 | 18 | 2 |
| recall-fiqa | positive_recall_returned | 0.311 | 0.291 | -0.021 | 0 | 18 | 2 |
| recall-fiqa | elapsed_seconds_mean | 10.6 | 11.2 | +0.6 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.147 | 0.153 | +0.006 | 2 | 17 | 1 |
| recall-nfcorpus | positive_recall_returned | 0.147 | 0.153 | +0.006 | 2 | 17 | 1 |
| recall-nfcorpus | elapsed_seconds_mean | 10.6 | 10.4 | -0.2 | | | |
| v2 | evidence_coverage_delivered | 0.926 | 0.947 | +0.021 | 2 | 44 | 1 |
| v2 | evidence_coverage_returned | 0.926 | 0.947 | +0.021 | 2 | 44 | 1 |
| v2 | source_recall_cited | 0.933 | 0.837 | -0.096 | 2 | 43 | 7 |
| v2 | elapsed_seconds_mean | 6.1 | 6.5 | +0.5 | | | |

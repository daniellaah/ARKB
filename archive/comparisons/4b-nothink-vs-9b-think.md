# Dev loop comparison: `acquisition-fix-4b-nothink` vs `cap-9b-think`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:9b think=True git=32e61925e2
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | completeness_returned | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 16.0 | 37.3 | +21.3 | | | |
| musique | answer_f1 | 0.083 | 0.084 | +0.001 | 5 | 2 | 3 |
| musique | answerability_correct | 0.600 | 0.750 | +0.150 | 4 | 15 | 1 |
| musique | support_f1 | 0.233 | 0.674 | +0.441 | 7 | 3 | 0 |
| musique | elapsed_seconds_mean | 11.3 | 43.3 | +32.0 | | | |
| recall-fiqa | positive_recall_delivered | 0.324 | 0.421 | +0.097 | 6 | 14 | 0 |
| recall-fiqa | positive_recall_returned | 0.324 | 0.421 | +0.097 | 6 | 14 | 0 |
| recall-fiqa | elapsed_seconds_mean | 13.5 | 42.0 | +28.5 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.170 | 0.194 | +0.024 | 4 | 15 | 1 |
| recall-nfcorpus | positive_recall_returned | 0.170 | 0.201 | +0.031 | 6 | 13 | 1 |
| recall-nfcorpus | elapsed_seconds_mean | 16.8 | 42.7 | +25.9 | | | |
| v2 | evidence_coverage_delivered | 0.947 | 0.968 | +0.021 | 1 | 46 | 0 |
| v2 | evidence_coverage_returned | 0.947 | 0.968 | +0.021 | 1 | 46 | 0 |
| v2 | source_recall_cited | 0.913 | 0.990 | +0.077 | 6 | 46 | 0 |
| v2 | elapsed_seconds_mean | 7.3 | 27.9 | +20.6 | | | |

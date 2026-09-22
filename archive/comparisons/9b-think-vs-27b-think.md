# Dev loop comparison: `cap-9b-think` vs `cap-27b-think`

A: product qwen3.5:9b think=True git=32e61925e2
B: product qwen3.5:27b think=True git=32e61925e2
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | completeness_returned | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 37.3 | 70.2 | +32.9 | | | |
| musique | answer_f1 | 0.084 | 0.069 | -0.015 | 3 | 2 | 5 |
| musique | answerability_correct | 0.750 | 0.650 | -0.100 | 1 | 16 | 3 |
| musique | support_f1 | 0.674 | 0.713 | +0.039 | 2 | 8 | 0 |
| musique | elapsed_seconds_mean | 43.3 | 119.8 | +76.5 | | | |
| recall-fiqa | positive_recall_delivered | 0.421 | 0.446 | +0.025 | 1 | 18 | 1 |
| recall-fiqa | positive_recall_returned | 0.421 | 0.446 | +0.025 | 1 | 18 | 1 |
| recall-fiqa | elapsed_seconds_mean | 42.0 | 90.8 | +48.9 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.194 | 0.191 | -0.003 | 3 | 15 | 2 |
| recall-nfcorpus | positive_recall_returned | 0.201 | 0.193 | -0.007 | 3 | 15 | 2 |
| recall-nfcorpus | elapsed_seconds_mean | 42.7 | 140.3 | +97.6 | | | |
| v2 | evidence_coverage_delivered | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | evidence_coverage_returned | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | source_recall_cited | 0.990 | 0.990 | +0.000 | 0 | 52 | 0 |
| v2 | elapsed_seconds_mean | 27.9 | 64.3 | +36.5 | | | |

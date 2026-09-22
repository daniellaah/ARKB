# Dev loop comparison: `acquisition-fix-4b-nothink` vs `cap-4b-think`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=True git=32e61925e2
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 1.000 | 0.917 | -0.083 | 0 | 22 | 2 |
| exact-nfcorpus | completeness_returned | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 16.0 | 35.5 | +19.6 | | | |
| musique | answer_f1 | 0.083 | 0.048 | -0.035 | 4 | 2 | 4 |
| musique | answerability_correct | 0.600 | 0.600 | +0.000 | 2 | 16 | 2 |
| musique | support_f1 | 0.233 | 0.567 | +0.333 | 6 | 3 | 1 |
| musique | elapsed_seconds_mean | 11.3 | 29.4 | +18.0 | | | |
| recall-fiqa | positive_recall_delivered | 0.324 | 0.336 | +0.012 | 2 | 17 | 1 |
| recall-fiqa | positive_recall_returned | 0.324 | 0.336 | +0.012 | 2 | 17 | 1 |
| recall-fiqa | elapsed_seconds_mean | 13.5 | 25.9 | +12.4 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.170 | 0.174 | +0.004 | 3 | 14 | 3 |
| recall-nfcorpus | positive_recall_returned | 0.170 | 0.177 | +0.007 | 4 | 13 | 3 |
| recall-nfcorpus | elapsed_seconds_mean | 16.8 | 26.0 | +9.2 | | | |
| v2 | evidence_coverage_delivered | 0.947 | 1.000 | +0.053 | 3 | 44 | 0 |
| v2 | evidence_coverage_returned | 0.947 | 1.000 | +0.053 | 3 | 44 | 0 |
| v2 | source_recall_cited | 0.913 | 0.923 | +0.010 | 5 | 44 | 3 |
| v2 | elapsed_seconds_mean | 7.3 | 15.1 | +7.8 | | | |

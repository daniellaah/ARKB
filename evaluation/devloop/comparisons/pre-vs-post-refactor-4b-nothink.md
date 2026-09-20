# Dev loop comparison: `acquisition-fix-4b-nothink` vs `refactor-check-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=af2a976b8e+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | completeness_returned | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 16.0 | 15.7 | -0.2 | | | |
| musique | answer_f1 | 0.083 | 0.086 | +0.003 | 2 | 7 | 1 |
| musique | answerability_correct | 0.600 | 0.600 | +0.000 | 0 | 20 | 0 |
| musique | support_f1 | 0.233 | 0.267 | +0.033 | 1 | 9 | 0 |
| musique | elapsed_seconds_mean | 11.3 | 12.1 | +0.7 | | | |
| recall-fiqa | positive_recall_delivered | 0.324 | 0.324 | +0.000 | 0 | 20 | 0 |
| recall-fiqa | positive_recall_returned | 0.324 | 0.324 | +0.000 | 0 | 20 | 0 |
| recall-fiqa | elapsed_seconds_mean | 13.5 | 13.0 | -0.5 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.170 | 0.172 | +0.002 | 1 | 19 | 0 |
| recall-nfcorpus | positive_recall_returned | 0.170 | 0.172 | +0.002 | 1 | 19 | 0 |
| recall-nfcorpus | elapsed_seconds_mean | 16.8 | 15.2 | -1.6 | | | |
| v2 | evidence_coverage_delivered | 0.947 | 0.968 | +0.021 | 1 | 46 | 0 |
| v2 | evidence_coverage_returned | 0.947 | 0.968 | +0.021 | 1 | 46 | 0 |
| v2 | source_recall_cited | 0.913 | 0.894 | -0.019 | 2 | 47 | 3 |
| v2 | elapsed_seconds_mean | 7.3 | 7.3 | -0.0 | | | |

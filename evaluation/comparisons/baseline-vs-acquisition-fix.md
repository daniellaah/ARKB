# Dev loop comparison: `baseline-4b-nothink` vs `acquisition-fix-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=befa4d457c+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.722 | 1.000 | +0.278 | 13 | 11 | 0 |
| exact-nfcorpus | completeness_returned | 0.722 | 1.000 | +0.278 | 13 | 11 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 8.4 | 16.0 | +7.6 | | | |
| musique | answer_f1 | 0.073 | 0.083 | +0.010 | 3 | 5 | 2 |
| musique | answerability_correct | 0.550 | 0.600 | +0.050 | 1 | 19 | 0 |
| musique | support_f1 | 0.267 | 0.233 | -0.033 | 1 | 7 | 2 |
| musique | elapsed_seconds_mean | 11.1 | 11.3 | +0.3 | | | |
| recall-fiqa | positive_recall_delivered | 0.311 | 0.324 | +0.012 | 1 | 19 | 0 |
| recall-fiqa | positive_recall_returned | 0.311 | 0.324 | +0.012 | 1 | 19 | 0 |
| recall-fiqa | elapsed_seconds_mean | 10.6 | 13.5 | +2.9 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.147 | 0.170 | +0.023 | 6 | 14 | 0 |
| recall-nfcorpus | positive_recall_returned | 0.147 | 0.170 | +0.023 | 6 | 14 | 0 |
| recall-nfcorpus | elapsed_seconds_mean | 10.6 | 16.8 | +6.3 | | | |
| v2 | evidence_coverage_delivered | 0.926 | 0.947 | +0.021 | 2 | 44 | 1 |
| v2 | evidence_coverage_returned | 0.926 | 0.947 | +0.021 | 2 | 44 | 1 |
| v2 | source_recall_cited | 0.933 | 0.913 | -0.019 | 0 | 51 | 1 |
| v2 | elapsed_seconds_mean | 6.1 | 7.3 | +1.2 | | | |

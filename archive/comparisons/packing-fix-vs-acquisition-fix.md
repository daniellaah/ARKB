# Dev loop comparison: `packing-fix-4b-nothink` vs `acquisition-fix-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=befa4d457c+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | completeness_returned | 1.000 | 1.000 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 13.5 | 16.0 | +2.5 | | | |
| musique | answer_f1 | 0.070 | 0.083 | +0.013 | 3 | 3 | 4 |
| musique | answerability_correct | 0.600 | 0.600 | +0.000 | 0 | 20 | 0 |
| musique | support_f1 | 0.252 | 0.233 | -0.019 | 1 | 8 | 1 |
| musique | elapsed_seconds_mean | 9.8 | 11.3 | +1.5 | | | |
| recall-fiqa | positive_recall_delivered | 0.311 | 0.324 | +0.012 | 2 | 17 | 1 |
| recall-fiqa | positive_recall_returned | 0.311 | 0.324 | +0.012 | 2 | 17 | 1 |
| recall-fiqa | elapsed_seconds_mean | 11.4 | 13.5 | +2.1 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.156 | 0.170 | +0.014 | 5 | 15 | 0 |
| recall-nfcorpus | positive_recall_returned | 0.156 | 0.170 | +0.014 | 5 | 15 | 0 |
| recall-nfcorpus | elapsed_seconds_mean | 10.8 | 16.8 | +6.1 | | | |
| v2 | evidence_coverage_delivered | 0.957 | 0.947 | -0.011 | 1 | 45 | 1 |
| v2 | evidence_coverage_returned | 0.957 | 0.947 | -0.011 | 1 | 45 | 1 |
| v2 | source_recall_cited | 0.846 | 0.913 | +0.067 | 5 | 47 | 0 |
| v2 | elapsed_seconds_mean | 6.4 | 7.3 | +0.9 | | | |

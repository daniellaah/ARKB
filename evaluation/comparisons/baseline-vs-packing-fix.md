# Dev loop comparison: `baseline-4b-nothink` vs `packing-fix-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=befa4d457c+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.722 | 1.000 | +0.278 | 13 | 11 | 0 |
| exact-nfcorpus | completeness_returned | 0.722 | 1.000 | +0.278 | 13 | 11 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 8.4 | 13.5 | +5.1 | | | |
| musique | answer_f1 | 0.073 | 0.070 | -0.003 | 4 | 3 | 3 |
| musique | answerability_correct | 0.550 | 0.600 | +0.050 | 1 | 19 | 0 |
| musique | support_f1 | 0.267 | 0.252 | -0.014 | 1 | 8 | 1 |
| musique | elapsed_seconds_mean | 11.1 | 9.8 | -1.2 | | | |
| recall-fiqa | positive_recall_delivered | 0.311 | 0.311 | +0.000 | 1 | 18 | 1 |
| recall-fiqa | positive_recall_returned | 0.311 | 0.311 | +0.000 | 1 | 18 | 1 |
| recall-fiqa | elapsed_seconds_mean | 10.6 | 11.4 | +0.8 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.147 | 0.156 | +0.009 | 2 | 18 | 0 |
| recall-nfcorpus | positive_recall_returned | 0.147 | 0.156 | +0.009 | 2 | 18 | 0 |
| recall-nfcorpus | elapsed_seconds_mean | 10.6 | 10.8 | +0.2 | | | |
| v2 | evidence_coverage_delivered | 0.926 | 0.957 | +0.032 | 2 | 44 | 1 |
| v2 | evidence_coverage_returned | 0.926 | 0.957 | +0.032 | 2 | 44 | 1 |
| v2 | source_recall_cited | 0.933 | 0.846 | -0.087 | 0 | 46 | 6 |
| v2 | elapsed_seconds_mean | 6.1 | 6.4 | +0.4 | | | |

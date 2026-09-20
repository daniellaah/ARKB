# Dev loop comparison: `aw-9b-think-F-H` vs `aw-9b-think-A-All`

A: contract:F-H qwen3.5:9b think=True git=942cdc4361+dirty
B: contract:A-All qwen3.5:9b think=True git=942cdc4361+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.057 | 0.875 | +0.818 | 21 | 1 | 2 |
| exact-nfcorpus | completeness_returned | 0.177 | 1.000 | +0.823 | 23 | 1 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 67.7 | 32.9 | -34.8 | | | |
| musique | answer_f1 | 0.250 | 0.600 | +0.350 | 4 | 6 | 0 |
| musique | answerability_correct | 0.350 | 0.650 | +0.300 | 6 | 14 | 0 |
| musique | support_f1 | 0.395 | 0.654 | +0.259 | 6 | 2 | 2 |
| musique | answer_f1_first_line | 0.250 | 0.600 | +0.350 | 4 | 6 | 0 |
| musique | answer_em_first_line | 0.200 | 0.600 | +0.400 | 4 | 6 | 0 |
| musique | elapsed_seconds_mean | 47.3 | 50.0 | +2.7 | | | |
| recall-fiqa | positive_recall_delivered | 0.462 | 0.400 | -0.062 | 1 | 15 | 4 |
| recall-fiqa | positive_recall_returned | 0.462 | 0.400 | -0.062 | 1 | 15 | 4 |
| recall-fiqa | elapsed_seconds_mean | 32.0 | 29.3 | -2.8 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.190 | 0.166 | -0.024 | 1 | 13 | 6 |
| recall-nfcorpus | positive_recall_returned | 0.190 | 0.175 | -0.015 | 2 | 12 | 6 |
| recall-nfcorpus | elapsed_seconds_mean | 44.9 | 31.1 | -13.8 | | | |
| v2 | evidence_coverage_delivered | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | evidence_coverage_returned | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | source_recall_cited | 0.567 | 0.894 | +0.327 | 22 | 25 | 5 |
| v2 | elapsed_seconds_mean | 39.6 | 14.6 | -25.0 | | | |

# Dev loop comparison: `aw-9b-think-F-S` vs `aw-9b-think-A-S`

A: contract:F-S qwen3.5:9b think=True git=942cdc4361+dirty
B: contract:A-S qwen3.5:9b think=True git=942cdc4361+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.016 | 0.167 | +0.151 | 9 | 15 | 0 |
| exact-nfcorpus | completeness_returned | 0.055 | 0.251 | +0.196 | 14 | 10 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 63.5 | 65.1 | +1.6 | | | |
| musique | answer_f1 | 0.325 | 0.675 | +0.350 | 4 | 6 | 0 |
| musique | answerability_correct | 0.500 | 0.700 | +0.200 | 5 | 14 | 1 |
| musique | support_f1 | 0.460 | 0.560 | +0.100 | 3 | 4 | 3 |
| musique | answer_f1_first_line | 0.325 | 0.675 | +0.350 | 4 | 6 | 0 |
| musique | answer_em_first_line | 0.300 | 0.500 | +0.200 | 2 | 8 | 0 |
| musique | elapsed_seconds_mean | 38.9 | 48.4 | +9.6 | | | |
| recall-fiqa | positive_recall_delivered | 0.505 | 0.400 | -0.105 | 1 | 14 | 5 |
| recall-fiqa | positive_recall_returned | 0.505 | 0.400 | -0.105 | 1 | 14 | 5 |
| recall-fiqa | elapsed_seconds_mean | 38.2 | 30.0 | -8.2 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.201 | 0.166 | -0.036 | 1 | 10 | 9 |
| recall-nfcorpus | positive_recall_returned | 0.201 | 0.166 | -0.036 | 1 | 10 | 9 |
| recall-nfcorpus | elapsed_seconds_mean | 42.7 | 32.4 | -10.3 | | | |
| v2 | evidence_coverage_delivered | 0.968 | 0.957 | -0.011 | 0 | 46 | 1 |
| v2 | evidence_coverage_returned | 0.968 | 0.957 | -0.011 | 0 | 46 | 1 |
| v2 | source_recall_cited | 0.712 | 0.856 | +0.144 | 14 | 31 | 7 |
| v2 | elapsed_seconds_mean | 32.3 | 15.9 | -16.4 | | | |

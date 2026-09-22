# Dev loop comparison: `aw-9b-think-A-S` vs `aw-9b-think-A-All`

A: contract:A-S qwen3.5:9b think=True git=942cdc4361+dirty
B: contract:A-All qwen3.5:9b think=True git=942cdc4361+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.167 | 0.875 | +0.708 | 21 | 1 | 2 |
| exact-nfcorpus | completeness_returned | 0.251 | 1.000 | +0.749 | 22 | 2 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 65.1 | 32.9 | -32.2 | | | |
| musique | answer_f1 | 0.675 | 0.600 | -0.075 | 0 | 9 | 1 |
| musique | answerability_correct | 0.700 | 0.650 | -0.050 | 3 | 13 | 4 |
| musique | support_f1 | 0.560 | 0.654 | +0.094 | 4 | 4 | 2 |
| musique | answer_f1_first_line | 0.675 | 0.600 | -0.075 | 0 | 9 | 1 |
| musique | answer_em_first_line | 0.500 | 0.600 | +0.100 | 1 | 9 | 0 |
| musique | elapsed_seconds_mean | 48.4 | 50.0 | +1.6 | | | |
| recall-fiqa | positive_recall_delivered | 0.400 | 0.400 | +0.000 | 2 | 15 | 3 |
| recall-fiqa | positive_recall_returned | 0.400 | 0.400 | +0.000 | 2 | 15 | 3 |
| recall-fiqa | elapsed_seconds_mean | 30.0 | 29.3 | -0.8 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.166 | 0.166 | +0.001 | 1 | 19 | 0 |
| recall-nfcorpus | positive_recall_returned | 0.166 | 0.175 | +0.010 | 2 | 18 | 0 |
| recall-nfcorpus | elapsed_seconds_mean | 32.4 | 31.1 | -1.3 | | | |
| v2 | evidence_coverage_delivered | 0.957 | 0.968 | +0.011 | 1 | 46 | 0 |
| v2 | evidence_coverage_returned | 0.957 | 0.968 | +0.011 | 1 | 46 | 0 |
| v2 | source_recall_cited | 0.856 | 0.894 | +0.038 | 5 | 43 | 4 |
| v2 | elapsed_seconds_mean | 15.9 | 14.6 | -1.2 | | | |

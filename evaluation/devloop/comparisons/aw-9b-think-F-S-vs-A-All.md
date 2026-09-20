# Dev loop comparison: `aw-9b-think-F-S` vs `aw-9b-think-A-All`

A: contract:F-S qwen3.5:9b think=True git=942cdc4361+dirty
B: contract:A-All qwen3.5:9b think=True git=942cdc4361+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.016 | 0.875 | +0.859 | 21 | 2 | 1 |
| exact-nfcorpus | completeness_returned | 0.055 | 1.000 | +0.945 | 24 | 0 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 63.5 | 32.9 | -30.6 | | | |
| musique | answer_f1 | 0.325 | 0.600 | +0.275 | 3 | 7 | 0 |
| musique | answerability_correct | 0.500 | 0.650 | +0.150 | 5 | 13 | 2 |
| musique | support_f1 | 0.460 | 0.654 | +0.194 | 4 | 3 | 3 |
| musique | answer_f1_first_line | 0.325 | 0.600 | +0.275 | 3 | 7 | 0 |
| musique | answer_em_first_line | 0.300 | 0.600 | +0.300 | 3 | 7 | 0 |
| musique | elapsed_seconds_mean | 38.9 | 50.0 | +11.1 | | | |
| recall-fiqa | positive_recall_delivered | 0.505 | 0.400 | -0.105 | 2 | 11 | 7 |
| recall-fiqa | positive_recall_returned | 0.505 | 0.400 | -0.105 | 2 | 11 | 7 |
| recall-fiqa | elapsed_seconds_mean | 38.2 | 29.3 | -9.0 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.201 | 0.166 | -0.035 | 1 | 10 | 9 |
| recall-nfcorpus | positive_recall_returned | 0.201 | 0.175 | -0.026 | 2 | 10 | 8 |
| recall-nfcorpus | elapsed_seconds_mean | 42.7 | 31.1 | -11.6 | | | |
| v2 | evidence_coverage_delivered | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | evidence_coverage_returned | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | source_recall_cited | 0.712 | 0.894 | +0.183 | 15 | 30 | 7 |
| v2 | elapsed_seconds_mean | 32.3 | 14.6 | -17.7 | | | |

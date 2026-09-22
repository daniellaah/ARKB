# Dev loop comparison: `aw-9b-think-F-H-2req` vs `aw-9b-think-A-All`

A: contract:F-H qwen3.5:9b think=True git=b2b3246ddf
B: contract:A-All qwen3.5:9b think=True git=942cdc4361+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.087 | 0.875 | +0.788 | 21 | 1 | 2 |
| exact-nfcorpus | completeness_returned | 0.177 | 1.000 | +0.823 | 23 | 1 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 66.8 | 32.9 | -33.9 | | | |
| musique | answer_f1 | 0.400 | 0.600 | +0.200 | 2 | 8 | 0 |
| musique | answerability_correct | 0.200 | 0.650 | +0.450 | 10 | 9 | 1 |
| musique | support_f1 | 0.069 | 0.654 | +0.585 | 9 | 0 | 1 |
| musique | answer_f1_first_line | 0.400 | 0.600 | +0.200 | 2 | 8 | 0 |
| musique | answer_em_first_line | 0.400 | 0.600 | +0.200 | 2 | 8 | 0 |
| musique | elapsed_seconds_mean | 54.9 | 50.0 | -4.9 | | | |
| recall-fiqa | positive_recall_delivered | 0.462 | 0.400 | -0.062 | 1 | 15 | 4 |
| recall-fiqa | positive_recall_returned | 0.462 | 0.400 | -0.062 | 1 | 15 | 4 |
| recall-fiqa | elapsed_seconds_mean | 45.4 | 29.3 | -16.1 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.190 | 0.166 | -0.024 | 1 | 13 | 6 |
| recall-nfcorpus | positive_recall_returned | 0.190 | 0.175 | -0.015 | 2 | 12 | 6 |
| recall-nfcorpus | elapsed_seconds_mean | 51.3 | 31.1 | -20.2 | | | |
| v2 | evidence_coverage_delivered | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | evidence_coverage_returned | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | source_recall_cited | 0.471 | 0.894 | +0.423 | 26 | 25 | 1 |
| v2 | elapsed_seconds_mean | 46.2 | 14.6 | -31.6 | | | |

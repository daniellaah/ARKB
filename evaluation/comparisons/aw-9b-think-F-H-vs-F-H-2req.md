# Dev loop comparison: `aw-9b-think-F-H` vs `aw-9b-think-F-H-2req`

A: contract:F-H qwen3.5:9b think=True git=942cdc4361+dirty
B: contract:F-H qwen3.5:9b think=True git=b2b3246ddf
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.057 | 0.087 | +0.030 | 4 | 18 | 2 |
| exact-nfcorpus | completeness_returned | 0.177 | 0.177 | +0.000 | 0 | 24 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 67.7 | 66.8 | -0.9 | | | |
| musique | answer_f1 | 0.250 | 0.400 | +0.150 | 2 | 7 | 1 |
| musique | answerability_correct | 0.350 | 0.200 | -0.150 | 1 | 15 | 4 |
| musique | support_f1 | 0.395 | 0.069 | -0.326 | 1 | 5 | 4 |
| musique | answer_f1_first_line | 0.250 | 0.400 | +0.150 | 2 | 7 | 1 |
| musique | answer_em_first_line | 0.200 | 0.400 | +0.200 | 2 | 8 | 0 |
| musique | elapsed_seconds_mean | 47.3 | 54.9 | +7.6 | | | |
| recall-fiqa | positive_recall_delivered | 0.462 | 0.462 | +0.000 | 0 | 20 | 0 |
| recall-fiqa | positive_recall_returned | 0.462 | 0.462 | +0.000 | 0 | 20 | 0 |
| recall-fiqa | elapsed_seconds_mean | 32.0 | 45.4 | +13.4 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.190 | 0.190 | +0.000 | 0 | 20 | 0 |
| recall-nfcorpus | positive_recall_returned | 0.190 | 0.190 | +0.000 | 0 | 20 | 0 |
| recall-nfcorpus | elapsed_seconds_mean | 44.9 | 51.3 | +6.3 | | | |
| v2 | evidence_coverage_delivered | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | evidence_coverage_returned | 0.968 | 0.968 | +0.000 | 0 | 47 | 0 |
| v2 | source_recall_cited | 0.567 | 0.471 | -0.096 | 9 | 29 | 14 |
| v2 | elapsed_seconds_mean | 39.6 | 46.2 | +6.6 | | | |

# Dev loop comparison: `aw-4b-nothink-F-H` vs `aw-4b-nothink-A-All`

A: contract:F-H qwen3.5:4b think=False git=942cdc4361+dirty
B: contract:A-All qwen3.5:4b think=False git=942cdc4361+dirty
shared scenarios: 144

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact-nfcorpus | completeness_cited | 0.114 | 1.000 | +0.886 | 23 | 1 | 0 |
| exact-nfcorpus | completeness_returned | 0.177 | 1.000 | +0.823 | 23 | 1 | 0 |
| exact-nfcorpus | elapsed_seconds_mean | 7.5 | 15.8 | +8.3 | | | |
| musique | answer_f1 | 0.483 | 0.181 | -0.303 | 2 | 2 | 6 |
| musique | answerability_correct | 0.450 | 0.550 | +0.100 | 6 | 10 | 4 |
| musique | support_f1 | 0.605 | 0.347 | -0.258 | 1 | 3 | 6 |
| musique | answer_f1_first_line | 0.483 | 0.181 | -0.303 | 2 | 2 | 6 |
| musique | answer_em_first_line | 0.400 | 0.100 | -0.300 | 1 | 5 | 4 |
| musique | elapsed_seconds_mean | 5.3 | 12.4 | +7.0 | | | |
| recall-fiqa | positive_recall_delivered | 0.462 | 0.415 | -0.047 | 2 | 14 | 4 |
| recall-fiqa | positive_recall_returned | 0.462 | 0.415 | -0.047 | 2 | 14 | 4 |
| recall-fiqa | elapsed_seconds_mean | 7.5 | 9.3 | +1.8 | | | |
| recall-nfcorpus | positive_recall_delivered | 0.190 | 0.141 | -0.049 | 2 | 11 | 7 |
| recall-nfcorpus | positive_recall_returned | 0.190 | 0.141 | -0.049 | 2 | 11 | 7 |
| recall-nfcorpus | elapsed_seconds_mean | 8.4 | 13.3 | +4.9 | | | |
| v2 | evidence_coverage_delivered | 0.968 | 0.894 | -0.074 | 0 | 43 | 4 |
| v2 | evidence_coverage_returned | 0.968 | 0.894 | -0.074 | 0 | 43 | 4 |
| v2 | source_recall_cited | 0.913 | 0.817 | -0.096 | 4 | 39 | 9 |
| v2 | elapsed_seconds_mean | 3.9 | 6.2 | +2.3 | | | |

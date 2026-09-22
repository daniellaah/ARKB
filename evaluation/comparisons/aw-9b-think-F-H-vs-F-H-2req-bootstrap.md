# Paired bootstrap over development-loop runs

Development material, single run per arm, nominal 95% percentile intervals, no multiplicity correction. Seed 20260912, 20000 resamples, stratified by slice.

- F-H: `aw-9b-think-F-H` (qwen3.5:9b, think=True, git=942cdc4361+dirty)
- F-H-2req: `aw-9b-think-F-H-2req` (qwen3.5:9b, think=True, git=b2b3246ddf)

| slice | metric | n | F-H | F-H-2req | contrast | diff | 95% CI | win/tie/loss |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| exact-nfcorpus | completeness_cited | 24 | 0.057 | 0.087 | F-H-2req minus F-H | +0.030 | [-0.021, +0.099] | 4/18/2 |
| exact-nfcorpus | complete_and_exact_cited | 24 | 0.000 | 0.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/24/0 |
| exact-nfcorpus | spurious_cited | 24 | 0.000 | 0.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/24/0 |
| exact-nfcorpus | elapsed_s | 24 | 67.693 | 66.823 | F-H-2req minus F-H | -0.871 | [-6.392, +5.093] | 10/0/14 |
| exact-nfcorpus | costs.model_requests | 24 | 1.000 | 1.375 | F-H-2req minus F-H | +0.375 | [+0.167, +0.583] | 9/15/0 |
| exact-nfcorpus | tool_calls | 24 | 1.000 | 1.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/24/0 |
| exact-nfcorpus | prompt_tokens | 24 | 8593.542 | 10440.333 | F-H-2req minus F-H | +1846.792 | [+488.039, +3256.432] | 14/0/10 |
| exact-nfcorpus | eval_tokens | 24 | 2612.250 | 3638.333 | F-H-2req minus F-H | +1026.083 | [+335.120, +1770.720] | 9/11/4 |
| musique | support_f1 | 10 | 0.395 | 0.069 | F-H-2req minus F-H | -0.326 | [-0.606, -0.080] | 1/5/4 |
| musique | answerability_correct | 20 | 0.350 | 0.200 | F-H-2req minus F-H | -0.150 | [-0.350, +0.050] | 1/15/4 |
| musique | answer_f1_first_line | 10 | 0.250 | 0.400 | F-H-2req minus F-H | +0.150 | [-0.100, +0.450] | 2/7/1 |
| musique | answer_em_first_line | 10 | 0.200 | 0.400 | F-H-2req minus F-H | +0.200 | [+0.000, +0.500] | 2/8/0 |
| musique | answer_f1 | 10 | 0.250 | 0.400 | F-H-2req minus F-H | +0.150 | [-0.100, +0.450] | 2/7/1 |
| musique | answer_em | 10 | 0.200 | 0.400 | F-H-2req minus F-H | +0.200 | [+0.000, +0.500] | 2/8/0 |
| musique | elapsed_s | 20 | 47.264 | 54.870 | F-H-2req minus F-H | +7.606 | [-0.989, +17.281] | 14/0/6 |
| musique | costs.model_requests | 20 | 1.000 | 1.500 | F-H-2req minus F-H | +0.500 | [+0.300, +0.700] | 10/10/0 |
| musique | tool_calls | 20 | 1.000 | 1.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| musique | prompt_tokens | 20 | 5433.700 | 7210.800 | F-H-2req minus F-H | +1777.100 | [+674.890, +2874.512] | 14/0/6 |
| musique | eval_tokens | 20 | 1898.800 | 3166.700 | F-H-2req minus F-H | +1267.900 | [+542.344, +2038.256] | 11/6/3 |
| recall-fiqa | positive_recall_delivered | 20 | 0.462 | 0.462 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-fiqa | positive_recall_returned | 20 | 0.462 | 0.462 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-fiqa | positive_recall_submitted | 20 | 0.462 | 0.462 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-fiqa | elapsed_s | 20 | 32.007 | 45.397 | F-H-2req minus F-H | +13.390 | [+7.955, +19.061] | 18/0/2 |
| recall-fiqa | costs.model_requests | 20 | 1.000 | 1.800 | F-H-2req minus F-H | +0.800 | [+0.600, +0.950] | 16/4/0 |
| recall-fiqa | tool_calls | 20 | 1.000 | 1.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-fiqa | prompt_tokens | 20 | 7665.000 | 12895.050 | F-H-2req minus F-H | +5230.050 | [+3819.697, +6481.851] | 18/0/2 |
| recall-fiqa | eval_tokens | 20 | 617.300 | 2310.750 | F-H-2req minus F-H | +1693.450 | [+1233.735, +2175.412] | 18/2/0 |
| recall-nfcorpus | positive_recall_delivered | 20 | 0.190 | 0.190 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-nfcorpus | positive_recall_returned | 20 | 0.190 | 0.190 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-nfcorpus | positive_recall_submitted | 20 | 0.190 | 0.190 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-nfcorpus | elapsed_s | 20 | 44.921 | 51.264 | F-H-2req minus F-H | +6.343 | [-1.787, +15.092] | 14/0/6 |
| recall-nfcorpus | costs.model_requests | 20 | 1.000 | 1.700 | F-H-2req minus F-H | +0.700 | [+0.500, +0.900] | 14/6/0 |
| recall-nfcorpus | tool_calls | 20 | 1.000 | 1.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/20/0 |
| recall-nfcorpus | prompt_tokens | 20 | 9452.250 | 14563.800 | F-H-2req minus F-H | +5111.550 | [+3373.252, +6696.511] | 16/0/4 |
| recall-nfcorpus | eval_tokens | 20 | 1148.550 | 2559.800 | F-H-2req minus F-H | +1411.250 | [+698.943, +2100.452] | 15/3/2 |
| v2 | evidence_coverage_delivered | 47 | 0.968 | 0.968 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/47/0 |
| v2 | source_recall_cited | 52 | 0.567 | 0.471 | F-H-2req minus F-H | -0.096 | [-0.269, +0.077] | 9/29/14 |
| v2 | behavior.answered | 60 | 0.383 | 0.333 | F-H-2req minus F-H | -0.050 | [-0.183, +0.083] | 7/43/10 |
| v2 | behavior.no_retrieval_respected | 4 | 0.000 | 0.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/4/0 |
| v2 | behavior.abstained | 4 | 0.500 | 0.250 | F-H-2req minus F-H | -0.250 | [-0.750, +0.000] | 0/3/1 |
| v2 | elapsed_s | 60 | 39.594 | 46.175 | F-H-2req minus F-H | +6.581 | [+0.338, +12.994] | 30/0/30 |
| v2 | costs.model_requests | 60 | 1.000 | 1.667 | F-H-2req minus F-H | +0.667 | [+0.550, +0.783] | 40/20/0 |
| v2 | tool_calls | 60 | 1.000 | 1.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/60/0 |
| v2 | prompt_tokens | 60 | 4238.233 | 5802.550 | F-H-2req minus F-H | +1564.317 | [+1075.958, +2050.920] | 42/0/18 |
| v2 | eval_tokens | 60 | 1319.017 | 2730.983 | F-H-2req minus F-H | +1411.967 | [+939.699, +1883.492] | 42/11/7 |
| all (stratified) | elapsed_s | 144 | 45.029 | 51.423 | F-H-2req minus F-H | +6.394 | [+3.018, +9.810] | 86/0/58 |
| all (stratified) | costs.model_requests | 144 | 1.000 | 1.618 | F-H-2req minus F-H | +0.618 | [+0.542, +0.694] | 89/55/0 |
| all (stratified) | tool_calls | 144 | 1.000 | 1.000 | F-H-2req minus F-H | +0.000 | [+0.000, +0.000] | 0/144/0 |
| all (stratified) | prompt_tokens | 144 | 6330.264 | 8973.014 | F-H-2req minus F-H | +2642.750 | [+2186.226, +3093.361] | 104/0/40 |
| all (stratified) | eval_tokens | 144 | 1493.944 | 2860.583 | F-H-2req minus F-H | +1366.639 | [+1087.793, +1650.021] | 95/33/16 |

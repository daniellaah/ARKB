# Paired bootstrap over development-loop runs

Development material, single run per arm, nominal 95% percentile intervals, no multiplicity correction. Seed 20260912, 20000 resamples, stratified by slice.

- F-H: `aw-4b-nothink-F-H` (qwen3.5:4b, think=False, git=942cdc4361+dirty)
- A-All: `aw-4b-nothink-A-All` (qwen3.5:4b, think=False, git=942cdc4361+dirty)

| slice | metric | n | F-H | A-All | contrast | diff | 95% CI | win/tie/loss |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| exact-nfcorpus | completeness_cited | 24 | 0.114 | 1.000 | A-All minus F-H | +0.886 | [+0.785, +0.959] | 23/1/0 |
| exact-nfcorpus | complete_and_exact_cited | 24 | 0.000 | 1.000 | A-All minus F-H | +1.000 | [+1.000, +1.000] | 24/0/0 |
| exact-nfcorpus | spurious_cited | 24 | 0.625 | 0.000 | A-All minus F-H | -0.625 | [-1.625, -0.083] | 0/19/5 |
| exact-nfcorpus | elapsed_s | 24 | 7.527 | 15.816 | A-All minus F-H | +8.289 | [+4.152, +12.854] | 22/0/2 |
| exact-nfcorpus | costs.model_requests | 24 | 1.000 | 3.000 | A-All minus F-H | +2.000 | [+2.000, +2.000] | 24/0/0 |
| exact-nfcorpus | tool_calls | 24 | 1.000 | 1.000 | A-All minus F-H | +0.000 | [+0.000, +0.000] | 0/24/0 |
| exact-nfcorpus | prompt_tokens | 24 | 7565.625 | 5577.750 | A-All minus F-H | -1987.875 | [-2748.055, -1119.517] | 4/0/20 |
| exact-nfcorpus | eval_tokens | 24 | 187.833 | 1262.833 | A-All minus F-H | +1075.000 | [+725.371, +1458.750] | 23/0/1 |
| musique | support_f1 | 10 | 0.605 | 0.347 | A-All minus F-H | -0.258 | [-0.492, +0.000] | 1/3/6 |
| musique | answerability_correct | 20 | 0.450 | 0.550 | A-All minus F-H | +0.100 | [-0.200, +0.400] | 6/10/4 |
| musique | answer_f1_first_line | 10 | 0.483 | 0.181 | A-All minus F-H | -0.303 | [-0.633, +0.077] | 2/2/6 |
| musique | answer_em_first_line | 10 | 0.400 | 0.100 | A-All minus F-H | -0.300 | [-0.700, +0.100] | 1/5/4 |
| musique | answer_f1 | 10 | 0.483 | 0.181 | A-All minus F-H | -0.303 | [-0.633, +0.077] | 2/2/6 |
| musique | answer_em | 10 | 0.400 | 0.100 | A-All minus F-H | -0.300 | [-0.700, +0.100] | 1/5/4 |
| musique | elapsed_s | 20 | 5.331 | 12.352 | A-All minus F-H | +7.021 | [+5.383, +8.728] | 19/0/1 |
| musique | costs.model_requests | 20 | 1.000 | 7.050 | A-All minus F-H | +6.050 | [+5.450, +6.600] | 20/0/0 |
| musique | tool_calls | 20 | 1.000 | 5.700 | A-All minus F-H | +4.700 | [+3.900, +5.400] | 20/0/0 |
| musique | prompt_tokens | 20 | 4715.050 | 26207.400 | A-All minus F-H | +21492.350 | [+16459.551, +26915.752] | 20/0/0 |
| musique | eval_tokens | 20 | 186.400 | 435.100 | A-All minus F-H | +248.700 | [+153.850, +324.500] | 18/0/2 |
| recall-fiqa | positive_recall_delivered | 20 | 0.462 | 0.415 | A-All minus F-H | -0.047 | [-0.187, +0.075] | 2/14/4 |
| recall-fiqa | positive_recall_returned | 20 | 0.462 | 0.415 | A-All minus F-H | -0.047 | [-0.187, +0.075] | 2/14/4 |
| recall-fiqa | positive_recall_submitted | 20 | 0.462 | 0.415 | A-All minus F-H | -0.047 | [-0.187, +0.075] | 2/14/4 |
| recall-fiqa | elapsed_s | 20 | 7.522 | 9.272 | A-All minus F-H | +1.750 | [-0.031, +3.665] | 12/0/8 |
| recall-fiqa | costs.model_requests | 20 | 1.000 | 3.850 | A-All minus F-H | +2.850 | [+2.050, +3.700] | 20/0/0 |
| recall-fiqa | tool_calls | 20 | 1.000 | 3.350 | A-All minus F-H | +2.350 | [+1.700, +3.100] | 20/0/0 |
| recall-fiqa | prompt_tokens | 20 | 6820.100 | 13479.750 | A-All minus F-H | +6659.650 | [+2219.934, +11880.949] | 14/0/6 |
| recall-fiqa | eval_tokens | 20 | 238.000 | 472.050 | A-All minus F-H | +234.050 | [+148.549, +315.401] | 19/0/1 |
| recall-nfcorpus | positive_recall_delivered | 20 | 0.190 | 0.141 | A-All minus F-H | -0.049 | [-0.127, +0.001] | 2/11/7 |
| recall-nfcorpus | positive_recall_returned | 20 | 0.190 | 0.141 | A-All minus F-H | -0.049 | [-0.127, +0.001] | 2/11/7 |
| recall-nfcorpus | positive_recall_submitted | 20 | 0.190 | 0.141 | A-All minus F-H | -0.049 | [-0.127, +0.001] | 2/11/7 |
| recall-nfcorpus | elapsed_s | 20 | 8.436 | 13.338 | A-All minus F-H | +4.903 | [+2.492, +7.719] | 17/0/3 |
| recall-nfcorpus | costs.model_requests | 20 | 1.000 | 4.800 | A-All minus F-H | +3.800 | [+3.150, +4.500] | 20/0/0 |
| recall-nfcorpus | tool_calls | 20 | 1.000 | 3.700 | A-All minus F-H | +2.700 | [+2.000, +3.450] | 19/1/0 |
| recall-nfcorpus | prompt_tokens | 20 | 8433.850 | 25089.800 | A-All minus F-H | +16655.950 | [+9544.196, +25547.454] | 19/0/1 |
| recall-nfcorpus | eval_tokens | 20 | 211.350 | 513.950 | A-All minus F-H | +302.600 | [+216.400, +392.601] | 19/0/1 |
| v2 | evidence_coverage_delivered | 47 | 0.968 | 0.894 | A-All minus F-H | -0.074 | [-0.160, -0.011] | 0/43/4 |
| v2 | source_recall_cited | 52 | 0.913 | 0.817 | A-All minus F-H | -0.096 | [-0.212, +0.019] | 4/39/9 |
| v2 | behavior.answered | 60 | 0.650 | 0.750 | A-All minus F-H | +0.100 | [-0.033, +0.233] | 12/42/6 |
| v2 | behavior.no_retrieval_respected | 4 | 0.000 | 0.750 | A-All minus F-H | +0.750 | [+0.250, +1.000] | 3/1/0 |
| v2 | behavior.abstained | 4 | 1.000 | 1.000 | A-All minus F-H | +0.000 | [+0.000, +0.000] | 0/4/0 |
| v2 | elapsed_s | 60 | 3.907 | 6.166 | A-All minus F-H | +2.260 | [+1.343, +3.202] | 46/0/14 |
| v2 | costs.model_requests | 60 | 1.000 | 4.217 | A-All minus F-H | +3.217 | [+2.783, +3.667] | 59/1/0 |
| v2 | tool_calls | 60 | 1.000 | 3.633 | A-All minus F-H | +2.633 | [+2.066, +3.233] | 49/8/3 |
| v2 | prompt_tokens | 60 | 3314.733 | 9976.250 | A-All minus F-H | +6661.517 | [+4782.425, +8733.505] | 51/0/9 |
| v2 | eval_tokens | 60 | 142.283 | 333.800 | A-All minus F-H | +191.517 | [+136.349, +242.450] | 57/0/3 |
| all (stratified) | elapsed_s | 144 | 5.839 | 10.061 | A-All minus F-H | +4.222 | [+3.280, +5.226] | 116/0/28 |
| all (stratified) | costs.model_requests | 144 | 1.000 | 4.438 | A-All minus F-H | +3.438 | [+3.194, +3.688] | 143/1/0 |
| all (stratified) | tool_calls | 144 | 1.000 | 3.451 | A-All minus F-H | +2.451 | [+2.153, +2.750] | 108/33/3 |
| all (stratified) | prompt_tokens | 144 | 5415.549 | 14083.194 | A-All minus F-H | +8667.646 | [+7034.065, +10418.535] | 108/0/36 |
| all (stratified) | eval_tokens | 144 | 178.889 | 546.931 | A-All minus F-H | +368.042 | [+301.706, +439.071] | 136/0/8 |

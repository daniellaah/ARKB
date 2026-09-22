# Paired bootstrap over development-loop runs

Development material, single run per arm, nominal 95% percentile intervals, no multiplicity correction. Seed 20260912, 20000 resamples, stratified by slice.

- F-S: `aw-long-9b-think-F-S` (qwen3.5:9b, think=True, git=942cdc4361+dirty)
- F-H: `aw-long-9b-think-F-H` (qwen3.5:9b, think=True, git=942cdc4361+dirty)
- A-S: `aw-long-9b-think-A-S` (qwen3.5:9b, think=True, git=942cdc4361+dirty)
- A-All: `aw-long-9b-think-A-All` (qwen3.5:9b, think=True, git=942cdc4361+dirty)

| slice | metric | n | F-S | F-H | A-S | A-All | contrast | diff | 95% CI | win/tie/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| long-browsecomp | positive_recall_delivered | 10 | 0.064 | 0.130 | 0.070 | 0.013 | A-All minus F-H | -0.118 | [-0.227, -0.027] | 1/3/6 |
| | | |  |  |  |  | A-All minus F-S | -0.052 | [-0.107, +0.000] | 0/7/3 |
| | | |  |  |  |  | A-S minus F-S | +0.005 | [-0.075, +0.087] | 2/6/2 |
| | | |  |  |  |  | A-All minus A-S | -0.057 | [-0.143, +0.000] | 0/8/2 |
| long-browsecomp | positive_recall_returned | 10 | 0.064 | 0.130 | 0.070 | 0.041 | A-All minus F-H | -0.089 | [-0.202, +0.000] | 1/5/4 |
| | | |  |  |  |  | A-All minus F-S | -0.023 | [-0.087, +0.030] | 1/7/2 |
| | | |  |  |  |  | A-S minus F-S | +0.005 | [-0.075, +0.087] | 2/6/2 |
| | | |  |  |  |  | A-All minus A-S | -0.029 | [-0.086, +0.000] | 0/9/1 |
| long-browsecomp | positive_recall_submitted | 10 | 0.064 | 0.130 | 0.070 | 0.013 | A-All minus F-H | -0.118 | [-0.227, -0.027] | 1/3/6 |
| | | |  |  |  |  | A-All minus F-S | -0.052 | [-0.107, +0.000] | 0/7/3 |
| | | |  |  |  |  | A-S minus F-S | +0.005 | [-0.075, +0.087] | 2/6/2 |
| | | |  |  |  |  | A-All minus A-S | -0.057 | [-0.143, +0.000] | 0/8/2 |
| long-browsecomp | elapsed_s | 10 | 57.304 | 61.892 | 41.296 | 39.048 | A-All minus F-H | -22.843 | [-38.293, -6.733] | 3/0/7 |
| | | |  |  |  |  | A-All minus F-S | -18.256 | [-32.333, -3.772] | 2/0/8 |
| | | |  |  |  |  | A-S minus F-S | -16.008 | [-28.612, -3.435] | 3/0/7 |
| | | |  |  |  |  | A-All minus A-S | -2.248 | [-5.337, +0.898] | 3/0/7 |
| long-browsecomp | costs.model_requests | 10 | 1.000 | 1.000 | 4.100 | 3.900 | A-All minus F-H | +2.900 | [+2.700, +3.000] | 10/0/0 |
| | | |  |  |  |  | A-All minus F-S | +2.900 | [+2.700, +3.000] | 10/0/0 |
| | | |  |  |  |  | A-S minus F-S | +3.100 | [+2.800, +3.400] | 10/0/0 |
| | | |  |  |  |  | A-All minus A-S | -0.200 | [-0.600, +0.200] | 1/6/3 |
| long-browsecomp | tool_calls | 10 | 1.000 | 1.000 | 3.200 | 3.200 | A-All minus F-H | +2.200 | [+2.000, +2.500] | 10/0/0 |
| | | |  |  |  |  | A-All minus F-S | +2.200 | [+2.000, +2.500] | 10/0/0 |
| | | |  |  |  |  | A-S minus F-S | +2.200 | [+2.000, +2.500] | 10/0/0 |
| | | |  |  |  |  | A-All minus A-S | +0.000 | [-0.300, +0.300] | 1/8/1 |
| long-browsecomp | prompt_tokens | 10 | 11119.800 | 11501.700 | 32024.500 | 31746.500 | A-All minus F-H | +20244.800 | [+18189.978, +22175.162] | 10/0/0 |
| | | |  |  |  |  | A-All minus F-S | +20626.700 | [+18112.630, +22876.715] | 10/0/0 |
| | | |  |  |  |  | A-S minus F-S | +20904.700 | [+17880.480, +24518.435] | 10/0/0 |
| | | |  |  |  |  | A-All minus A-S | -278.000 | [-3957.012, +3327.680] | 7/0/3 |
| long-browsecomp | eval_tokens | 10 | 2075.900 | 1682.800 | 763.000 | 622.100 | A-All minus F-H | -1060.700 | [-2349.503, +210.607] | 6/0/4 |
| | | |  |  |  |  | A-All minus F-S | -1453.800 | [-2709.100, -190.988] | 5/0/5 |
| | | |  |  |  |  | A-S minus F-S | -1312.900 | [-2495.800, -155.200] | 5/0/5 |
| | | |  |  |  |  | A-All minus A-S | -140.900 | [-348.902, +53.300] | 3/0/7 |

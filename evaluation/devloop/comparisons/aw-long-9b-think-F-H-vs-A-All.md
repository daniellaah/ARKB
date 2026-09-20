# Dev loop comparison: `aw-long-9b-think-F-H` vs `aw-long-9b-think-A-All`

A: contract:F-H qwen3.5:9b think=True git=942cdc4361+dirty
B: contract:A-All qwen3.5:9b think=True git=942cdc4361+dirty
shared scenarios: 10

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long-browsecomp | positive_recall_delivered | 0.130 | 0.013 | -0.118 | 1 | 3 | 6 |
| long-browsecomp | positive_recall_returned | 0.130 | 0.041 | -0.089 | 1 | 5 | 4 |
| long-browsecomp | elapsed_seconds_mean | 61.9 | 39.0 | -22.8 | | | |

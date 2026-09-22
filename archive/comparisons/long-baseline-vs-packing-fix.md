# Dev loop comparison: `long-baseline-4b-nothink` vs `long-packing-fix-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=befa4d457c+dirty
shared scenarios: 10

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long-browsecomp | positive_recall_delivered | 0.154 | 0.166 | +0.013 | 2 | 7 | 1 |
| long-browsecomp | positive_recall_returned | 0.225 | 0.166 | -0.059 | 2 | 5 | 3 |
| long-browsecomp | elapsed_seconds_mean | 20.0 | 26.4 | +6.3 | | | |

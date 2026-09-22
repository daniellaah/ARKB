# Dev loop comparison: `long-packing-fix-4b-nothink` vs `long-acquisition-fix-4b-nothink`

A: product qwen3.5:4b think=False git=befa4d457c+dirty
B: product qwen3.5:4b think=False git=befa4d457c+dirty
shared scenarios: 10

| slice | metric | A | B | delta | wins | ties | losses |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long-browsecomp | positive_recall_delivered | 0.166 | 0.166 | +0.000 | 2 | 7 | 1 |
| long-browsecomp | positive_recall_returned | 0.166 | 0.166 | +0.000 | 2 | 7 | 1 |
| long-browsecomp | elapsed_seconds_mean | 26.4 | 25.5 | -0.9 | | | |

# Common development tasks. Run from the repository root.
PY := .venv/bin/python
EVAL := $(PY) -m evaluation.run

.PHONY: test test-all lint format devset eval rescore compare status

test:            ## deterministic suite (no models or services)
	$(PY) -m pytest -q -m 'not integration'

test-all:        ## including service integration tests (needs Ollama and Qdrant)
	ARKB_RUN_MODEL_TESTS=1 $(PY) -m pytest -q

lint:            ## pyflakes-level checks
	.venv/bin/ruff check --select F,E9 src evaluation tests

format:          ## format the modules listed in FILES, e.g. make format FILES=src/arkb/agent/loop.py
	.venv/bin/ruff format $(FILES)

devset:          ## rebuild the development set and the v2 corpus index
	$(PY) -m evaluation.devset.build --index

eval:            ## run the product agent on the devset, e.g. make eval LABEL=my-change MODEL=qwen3.5:9b THINK=--think SLICES=core
	$(EVAL) run --label $(LABEL) --model $(or $(MODEL),qwen3.5:9b) $(THINK) --slices $(or $(SLICES),core)

rescore:         ## recompute the scores of a saved run, e.g. make rescore LABEL=my-change
	$(EVAL) rescore evaluation/results/$(LABEL)

compare:         ## paired comparison of two runs, e.g. make compare A=baseline B=my-change
	$(PY) -m evaluation.compare --run $(A)=evaluation/results/$(A) --run $(B)=evaluation/results/$(B) --output evaluation/comparisons/$(A)-vs-$(B).md

status:          ## durable evaluation handoff record
	$(PY) -c "import json; d=json.load(open('evaluation/results/agentic-tools-v1/orchestration-status.json')); print(d['phase']); print(d['last_action'])"

help:
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## /  -  /'

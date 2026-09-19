# Common development tasks. Run from the repository root.
PY := .venv/bin/python
DEVLOOP := $(PY) -m evaluation.devloop.run

.PHONY: test test-all lint format devset devloop compare index-v2 status

test:            ## deterministic suite (no models or services)
	$(PY) -m pytest -q -m 'not integration'

test-all:        ## including service integration tests (needs Ollama and Qdrant)
	ARKB_RUN_MODEL_TESTS=1 $(PY) -m pytest -q

lint:            ## pyflakes-level checks
	.venv/bin/ruff check --select F,E9 src evaluation tests

format:          ## format the modules listed in FILES, e.g. make format FILES=src/arkb/agent/loop.py
	.venv/bin/ruff format $(FILES)

devset:          ## rebuild the development set and the v2 corpus index
	$(PY) -m evaluation.devloop.build --index

devloop:         ## run the development loop, e.g. make devloop LABEL=my-change MODEL=qwen3.5:4b THINK=
	$(DEVLOOP) run --output evaluation/results/devloop/$(LABEL) --label $(LABEL) --model $(or $(MODEL),qwen3.5:4b) $(THINK)

compare:         ## compare two runs, e.g. make compare A=baseline-4b-nothink B=my-change
	$(DEVLOOP) compare evaluation/results/devloop/$(A) evaluation/results/devloop/$(B) --output evaluation/devloop/comparisons/$(A)-vs-$(B).md

status:          ## durable evaluation handoff record
	$(PY) -c "import json; d=json.load(open('evaluation/results/agentic-tools-v1/orchestration-status.json')); print(d['phase']); print(d['last_action'])"

help:
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## /  -  /'

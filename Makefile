# RAG hybrid-search — developer entrypoints (targets per plan §7.6).
# Most targets beyond `install`/`test`/`lint` invoke modules that land in later
# phases; they are wired here so the workflow is ready as each phase is built.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help install test lint eval eval-smoke seed run-api run-ui up down

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with the dev tools (ruff, pytest).
	$(PIP) install -e ".[dev]"

lint: ## Lint with ruff.
	$(PYTHON) -m ruff check .

test: ## Run the test suite (succeeds even with zero tests).
	$(PYTHON) -m pytest

eval: ## Run the full evaluation suite -> report (Phase 4).
	$(PYTHON) eval/run_eval.py

eval-smoke: ## Fast eval against mocks, safe for CI (no paid API calls) (Phase 4).
	$(PYTHON) eval/run_eval.py --smoke

seed: ## Ingest the sample corpus (Phase 5).
	$(PYTHON) scripts/seed.py

run-api: ## Run the FastAPI app locally (Phase 5).
	$(PYTHON) -m uvicorn rag.api.main:app --reload

run-ui: ## Run the Streamlit UI locally (Phase 5, V1).
	$(PYTHON) -m streamlit run ui/app.py

up: ## Start the stack with docker-compose (Phase 5).
	docker compose up -d

down: ## Stop the docker-compose stack (Phase 5).
	docker compose down

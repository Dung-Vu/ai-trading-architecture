# =============================================================================
# AI Trading Architecture — Makefile
# =============================================================================
SHELL := /bin/bash
PYTHON := python3
PIP := $(PYTHON) -m pip
COMPOSE := docker compose -f docker-compose.prod.yml

.PHONY: setup docker-up docker-down docker-logs docker-build \
        run-data run-dryrun run-ai run-dashboard \
        test lint format clean \
        help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup: ## Create venv and install all dependencies
	$(PYTHON) -m venv venv
	. venv/bin/activate && $(PIP) install --upgrade pip
	. venv/bin/activate && $(PIP) install -r requirements.txt
	@echo "✓  Virtual environment created and dependencies installed."

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-build: ## Build production Docker image
	$(COMPOSE) build

docker-up: ## Start all production services (detached)
	$(COMPOSE) up -d
	@echo "✓  Services started.  Run 'make docker-logs' to follow."

docker-down: ## Stop and remove all production containers
	$(COMPOSE) down

docker-logs: ## Follow logs from the trading app container
	$(COMPOSE) logs -f app

docker-status: ## Show status of all containers
	$(COMPOSE) ps

# ---------------------------------------------------------------------------
# Run commands
# ---------------------------------------------------------------------------
run-data: ## Run the data pipeline
	$(PYTHON) -m src.main --data-pipeline

run-dryrun: ## Run the bot in dry-run (paper trading) mode
	$(PYTHON) -m src.main --mode dryrun

run-ai: ## Run the AI debate strategy
	$(PYTHON) -m src.main_ai --strategy ai_debate

run-dashboard: ## Launch the Streamlit dashboard
	streamlit run src/dashboard.py

# ---------------------------------------------------------------------------
# Testing & Linting
# ---------------------------------------------------------------------------
test: ## Run pytest
	$(PYTHON) -m pytest tests/ -v

lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/

format: ## Auto-format code with ruff + black
	$(PYTHON) -m ruff format src/
	$(PYTHON) -m black src/

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean: ## Remove __pycache__, .pyc, build artifacts, and venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf venv/ .venv/ env/
	@echo "✓  Cleaned."

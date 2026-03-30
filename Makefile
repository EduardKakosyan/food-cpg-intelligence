.PHONY: install lint format typecheck test test-cov clean

install: ## Install all dependencies (including dev)
	uv sync --extra dev
	uv run pre-commit install

lint: ## Run ruff linter with auto-fix
	uv run ruff check src/ tests/ --fix

format: ## Run ruff formatter
	uv run ruff format src/ tests/

typecheck: ## Run mypy type checking
	uv run mypy src/

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov --cov-report=html

check: lint format typecheck test ## Run all checks

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info .mypy_cache .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help

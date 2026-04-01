# CLAUDE.md

## Project Overview

Food & CPG Product Intelligence Platform — a pure-Python domain-specific Q&A system with RAG and fine-tuned LLM. See `docs/food-cpg-intelligence-project-brief-v2.docx` for the full brief.

## Tech Stack

- **Python 3.11+** with **uv** for environment/dependency management
- **Unsloth** for QLoRA fine-tuning (Qwen 3.5 9B primary; Qwen 3.5 4B, Gemma 3 4B alternatives)
- **ChromaDB / Qdrant** for vector storage (evaluation pending)
- **FastAPI** for serving, **Typer** for CLI
- **RAGAS** + Claude (LLM-as-judge) for evaluation
- **Ruff** for linting/formatting, **mypy** for type checking, **pytest** for tests

## Commands

```bash
uv sync --extra dev          # Install all deps
uv run pytest                # Run tests
uv run ruff check src/ tests/ --fix  # Lint
uv run ruff format src/ tests/       # Format
uv run mypy src/             # Type check
uv run fcpg version          # CLI
```

## Code Conventions

- **src layout**: all source code under `src/food_cpg_intelligence/`
- **Config via env**: use `food_cpg_intelligence.config.settings` — all config comes from env vars with `FCPG_` prefix
- **Structured logging**: use `structlog` — no bare `print()` statements
- **Type hints**: all public functions must be typed; mypy strict mode is enforced
- **Import order**: managed by ruff isort — stdlib, third-party, first-party
- **Line length**: 100 chars
- **Tests**: mirror the `src/` structure under `tests/unit/` and `tests/integration/`

## Architecture Constraints

- Pure Python — no Go, no cross-language components
- QLoRA only (no full fine-tuning)
- Local-first development targeting Apple Silicon M4 Pro (48GB)
- Evaluation framework must be established before any fine-tuning
- Vector store must be evaluated (at least two options) before committing

# Food & CPG Product Intelligence Platform

Domain-specific Q&A system for food products, ingredients, nutritional data, and market trends — powered by a fine-tuned LLM with retrieval-augmented generation (RAG).

## Overview

This platform enables natural-language queries about food and CPG products, grounded in real data from Open Food Facts, USDA FoodData Central, and CFIA recalls. It combines a QLoRA fine-tuned model (via Unsloth) with a vector-based retrieval layer and a three-layer evaluation framework.

**Key capabilities:**

- Product discovery, comparative analysis, and ingredient interpretation
- Trend and category analysis across Canadian food products
- Hallucination-resistant responses with citation grounding
- Rigorous evaluation: gold standard test set, RAGAS metrics, LLM-as-judge

## Project Structure

```
food-cpg-intelligence/
├── src/food_cpg_intelligence/   # Main package (src layout)
│   ├── data/                    # Ingestion, normalization, embedding
│   ├── evaluation/              # Gold standard, RAGAS, LLM-as-judge
│   ├── training/                # Synthetic data generation, fine-tuning
│   ├── rag/                     # Retrieval-augmented generation pipeline
│   ├── api/                     # FastAPI serving layer
│   ├── config.py                # Pydantic settings from env
│   └── cli.py                   # Typer CLI entrypoint
├── tests/                       # Unit and integration tests
├── data/                        # Local data (not committed)
│   ├── raw/                     # Source datasets
│   ├── processed/               # Normalized data
│   ├── embeddings/              # Vector store files
│   ├── training/                # Synthetic Q&A pairs (JSONL)
│   └── evaluation/              # Gold standard test set
├── models/                      # LoRA adapters & checkpoints (not committed)
├── notebooks/                   # Exploration & analysis
├── configs/                     # Training & pipeline configs
└── docs/                        # Project briefs & documentation
```

## Setup

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone and enter the repo
git clone https://github.com/EduardKakosyan/food-cpg-intelligence.git
cd food-cpg-intelligence

# Install all dependencies (including dev)
uv sync --extra dev

# Set up pre-commit hooks
uv run pre-commit install

# Copy environment config
cp .env.example .env
# Edit .env with your API keys
```

## Usage

```bash
# Run the CLI
uv run fcpg version

# Run tests
uv run pytest

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run mypy src/

# Start the API server (once built)
uv run uvicorn food_cpg_intelligence.api:app --reload
```

## Development

- **Linting/formatting:** Ruff (configured in `pyproject.toml`)
- **Type checking:** mypy (strict mode)
- **Testing:** pytest with coverage
- **Pre-commit:** trailing whitespace, YAML/TOML validation, ruff lint + format
- **All commands** run through `uv run` to use the project virtualenv

## Delivery Phases

| Phase | Description |
|-------|-------------|
| 1 | Data Foundation — ingestion, normalization, embedding, vector store evaluation |
| 2 | Evaluation Foundation — gold standard test set, RAGAS, LLM-as-judge baseline |
| 3 | Training Data — synthetic Q&A generation and curation |
| 4 | Model Fine-Tuning — Unsloth QLoRA on Mistral 7B / Llama 3.1 8B |
| 5 | RAG Pipeline — end-to-end query-to-answer with citations |
| 6 | Cloud Deployment — serving, retraining automation |

## License

Internal R&D — AI-First Consulting Inc.

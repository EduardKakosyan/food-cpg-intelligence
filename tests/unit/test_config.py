"""Tests for application config."""

from pathlib import Path

from food_cpg_intelligence.config import Settings


def test_default_settings() -> None:
    s = Settings()
    assert s.data_dir == "data"
    assert s.vector_store_backend in ("chromadb", "qdrant")


def test_environment_defaults_to_local() -> None:
    s = Settings()
    assert s.environment == "local"


def test_data_path_defaults() -> None:
    s = Settings()
    assert s.raw_data_dir == "data/raw/SKUFood AI Project"
    assert s.newsletters_dir == "data/raw/SKUFood AI Project/SKUFood Newsletters"
    assert s.processed_dir == "data/processed"
    assert s.training_dir == "data/training"
    assert s.evaluation_dir == "data/evaluation"
    assert s.embeddings_dir == "data/embeddings"


def test_model_defaults() -> None:
    s = Settings()
    assert s.base_model == "Qwen/Qwen3.5-9B"
    assert s.claude_model == "claude-sonnet-4-6"
    assert s.ollama_model == ""
    assert s.lora_rank == 16
    assert s.lora_alpha == 32


def test_resolve_path_local() -> None:
    s = Settings(environment="local")
    result = s.resolve_path("data/processed")
    assert isinstance(result, Path)
    assert result.is_absolute()
    assert str(result).endswith("data/processed")


def test_resolve_path_colab() -> None:
    s = Settings(environment="colab", colab_base_path="/content/drive/MyDrive/fcpg")
    result = s.resolve_path("data/processed")
    assert result == Path("/content/drive/MyDrive/fcpg/data/processed")

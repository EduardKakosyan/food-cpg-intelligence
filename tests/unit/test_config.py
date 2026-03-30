"""Smoke tests for application config."""

from food_cpg_intelligence.config import Settings


def test_default_settings() -> None:
    s = Settings()
    assert s.data_dir == "data"
    assert s.vector_store_backend in ("chromadb", "qdrant")

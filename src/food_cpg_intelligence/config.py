"""Application configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings — populated from .env or environment variables."""

    model_config = {"env_prefix": "FCPG_", "env_file": ".env", "extra": "ignore"}

    # --- Environment ---
    environment: Literal["local", "colab"] = "local"

    # --- Anthropic (evaluation / synthetic data generation) ---
    anthropic_api_key: str = ""

    # --- Paths (local defaults — override via env for Colab) ---
    data_dir: str = "data"
    raw_data_dir: str = "data/raw/SKUFood AI Project"
    newsletters_dir: str = "data/raw/SKUFood AI Project/SKUFood Newsletters"
    processed_dir: str = "data/processed"
    training_dir: str = "data/training"
    evaluation_dir: str = "data/evaluation"
    embeddings_dir: str = "data/embeddings"
    models_dir: str = "models"

    # --- Colab ---
    colab_base_path: str = "/content/drive/MyDrive/food-cpg-intelligence"

    # --- Embedding ---
    embedding_model: str = "nomic-ai/nomic-embed-text-v2-moe"
    embedding_dim: int = 768

    # --- Vector Store ---
    vector_store_backend: str = "chromadb"  # "chromadb" | "qdrant"

    # --- Fine-tuning (MLX LoRA on Apple Silicon) ---
    base_model: str = "Qwen/Qwen3.5-9B"
    lora_rank: int = 16
    lora_alpha: int = 32
    adapter_dir: str = "models/adapters/skufood"
    gguf_dir: str = "models/gguf"
    training_config_path: str = "configs/training.yaml"

    # --- Models (inference) ---
    claude_model: str = "claude-sonnet-4-6"
    ollama_model: str = "skufood-9b-v2"

    # --- Serving ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Audio ingestion ---
    audio_raw_dir: str = "data/raw/audio"
    audio_processed_dir: str = "data/processed/audio"
    audio_cache_dir: str = "models/audio_cache"
    audio_config_path: str = "configs/audio.yaml"
    peter_reference_clip: str = "data/raw/audio/peter_reference.wav"
    huggingface_token: str = ""

    def resolve_path(self, relative: str) -> Path:
        """Return absolute path, adjusting base for local vs. Colab environment."""
        if self.environment == "colab":
            return Path(self.colab_base_path) / relative
        return Path(relative).resolve()


settings = Settings()

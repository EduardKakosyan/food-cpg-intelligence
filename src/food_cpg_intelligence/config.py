"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings — populated from .env or environment variables."""

    model_config = {"env_prefix": "FCPG_", "env_file": ".env", "extra": "ignore"}

    # --- Anthropic (evaluation / synthetic data generation) ---
    anthropic_api_key: str = ""

    # --- Paths ---
    data_dir: str = "data"
    models_dir: str = "models"

    # --- Embedding ---
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    embedding_dim: int = 768

    # --- Vector Store ---
    vector_store_backend: str = "chromadb"  # "chromadb" | "qdrant"

    # --- Fine-tuning ---
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    lora_rank: int = 16
    lora_alpha: int = 32

    # --- Serving ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()

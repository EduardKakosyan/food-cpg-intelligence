"""Pydantic models for MLX LoRA training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class LoraConfig(BaseModel, frozen=True):
    """LoRA adapter hyperparameters."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    layers: int = 16

    @field_validator("rank")
    @classmethod
    def _rank_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"rank must be >= 1, got {v}")
        return v

    @field_validator("dropout")
    @classmethod
    def _dropout_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {v}")
        return v


class TrainingConfig(BaseModel, frozen=True):
    """Full MLX LoRA training configuration."""

    # Model
    model: str = "Qwen/Qwen3.5-9B"

    # LoRA
    lora: LoraConfig = Field(default_factory=LoraConfig)

    # Training
    batch_size: int = 2
    grad_accumulation: int = 4
    iters: int = 600
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    warmup_steps: int = 50
    grad_checkpoint: bool = False
    seed: int = 42

    # Data paths
    data_dir: str = "data/training/mlx"
    adapter_dir: str = "models/adapters/skufood"

    # Reporting
    steps_per_report: int = 10
    steps_per_eval: int = 50
    save_every: int = 100
    val_batches: int = 25

    @field_validator("learning_rate")
    @classmethod
    def _lr_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"learning_rate must be > 0, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def _batch_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"batch_size must be >= 1, got {v}")
        return v

    def to_mlx_dict(self) -> dict[str, Any]:
        """Serialize to the dict structure mlx_lm.lora expects."""
        return {
            "model": self.model,
            "train": True,
            "data": self.data_dir,
            "seed": self.seed,
            "lora_layers": self.lora.layers,
            "lora_parameters": {
                "rank": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "scale": self.lora.alpha / self.lora.rank,
            },
            "batch_size": self.batch_size,
            "iters": self.iters,
            "val_batches": self.val_batches,
            "learning_rate": self.learning_rate,
            "steps_per_report": self.steps_per_report,
            "steps_per_eval": self.steps_per_eval,
            "adapter_path": self.adapter_dir,
            "save_every": self.save_every,
            "max_seq_length": self.max_seq_length,
            "grad_checkpoint": self.grad_checkpoint,
        }

    def to_mlx_yaml(self) -> str:
        """Serialize to YAML string for mlx_lm.lora --config."""
        result: str = yaml.dump(self.to_mlx_dict(), default_flow_style=False, sort_keys=False)
        return result


class ExportConfig(BaseModel, frozen=True):
    """Configuration for model export (fuse + GGUF + Ollama)."""

    base_model: str = "Qwen/Qwen3.5-9B"
    adapter_path: str = "models/adapters/skufood"
    fused_model_path: str = "models/fused/skufood"
    gguf_path: str = "models/gguf/skufood-9b.gguf"
    quantization: str = "q4_k_m"
    ollama_model_name: str = "skufood-9b"


def load_training_config(path: Path) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Flatten nested 'lora' and 'training' sections if present
    if "lora" in raw and isinstance(raw["lora"], dict):
        lora_data = raw.pop("lora")
        raw["lora"] = LoraConfig(**lora_data)

    return TrainingConfig(**raw)

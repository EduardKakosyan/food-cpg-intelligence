"""Tests for training config Pydantic models."""

from pathlib import Path

import pytest

from food_cpg_intelligence.training.config_models import (
    ExportConfig,
    LoraConfig,
    TrainingConfig,
    load_training_config,
)


def test_lora_config_defaults() -> None:
    cfg = LoraConfig()
    assert cfg.rank == 16
    assert cfg.alpha == 32
    assert cfg.dropout == 0.05
    assert cfg.layers == 16


def test_lora_config_rank_validation() -> None:
    with pytest.raises(ValueError, match="rank must be >= 1"):
        LoraConfig(rank=0)


def test_lora_config_dropout_validation() -> None:
    with pytest.raises(ValueError, match="dropout must be"):
        LoraConfig(dropout=1.0)
    with pytest.raises(ValueError, match="dropout must be"):
        LoraConfig(dropout=-0.1)


def test_training_config_defaults() -> None:
    cfg = TrainingConfig()
    assert cfg.model == "Qwen/Qwen3.5-9B"
    assert cfg.batch_size == 2
    assert cfg.iters == 600
    assert cfg.learning_rate == 2e-5


def test_training_config_lr_validation() -> None:
    with pytest.raises(ValueError, match="learning_rate must be > 0"):
        TrainingConfig(learning_rate=0.0)


def test_training_config_batch_validation() -> None:
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        TrainingConfig(batch_size=0)


def test_to_mlx_dict() -> None:
    cfg = TrainingConfig()
    d = cfg.to_mlx_dict()
    assert d["model"] == "Qwen/Qwen3.5-9B"
    assert d["train"] is True
    assert d["lora_layers"] == 16
    assert d["lora_parameters"]["rank"] == 16
    assert d["lora_parameters"]["scale"] == 2.0  # alpha / rank = 32 / 16
    assert d["batch_size"] == 2
    assert d["iters"] == 600


def test_to_mlx_yaml() -> None:
    cfg = TrainingConfig()
    yaml_str = cfg.to_mlx_yaml()
    assert "model:" in yaml_str
    assert "lora_layers:" in yaml_str
    assert "batch_size:" in yaml_str


def test_export_config_defaults() -> None:
    cfg = ExportConfig()
    assert cfg.quantization == "q4_k_m"
    assert cfg.ollama_model_name == "skufood-9b"


def test_load_training_config(tmp_path: Path) -> None:
    yaml_content = """
model: "Qwen/Qwen3.5-4B"
lora:
  rank: 8
  alpha: 16
  dropout: 0.1
  layers: 8
batch_size: 1
iters: 300
learning_rate: 1.0e-5
"""
    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml_content)

    cfg = load_training_config(cfg_path)
    assert cfg.model == "Qwen/Qwen3.5-4B"
    assert cfg.lora.rank == 8
    assert cfg.batch_size == 1
    assert cfg.iters == 300

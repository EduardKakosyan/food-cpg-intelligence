"""Tests for MLX trainer orchestration."""

from pathlib import Path

import pytest

from food_cpg_intelligence.training.config_models import TrainingConfig
from food_cpg_intelligence.training.mlx_trainer import (
    check_memory_budget,
    run_finetune,
    validate_data_dir,
)


def test_check_memory_budget_qwen9b() -> None:
    result = check_memory_budget("Qwen/Qwen3.5-9B")
    assert result["model"] == "Qwen/Qwen3.5-9B"
    assert result["estimated_gb"] > 0
    assert isinstance(result["fits"], bool)


def test_check_memory_budget_unknown_model() -> None:
    result = check_memory_budget("unknown/model-99B")
    assert result["estimated_gb"] > 0  # uses default estimate


def test_validate_data_dir_valid(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text('{"messages": []}\n')
    (tmp_path / "valid.jsonl").write_text('{"messages": []}\n')
    validate_data_dir(tmp_path)  # should not raise


def test_validate_data_dir_missing_train(tmp_path: Path) -> None:
    (tmp_path / "valid.jsonl").write_text('{"messages": []}\n')
    with pytest.raises(FileNotFoundError, match=r"train\.jsonl"):
        validate_data_dir(tmp_path)


def test_validate_data_dir_empty_train(tmp_path: Path) -> None:
    (tmp_path / "train.jsonl").write_text("")
    (tmp_path / "valid.jsonl").write_text('{"messages": []}\n')
    with pytest.raises(ValueError, match="empty"):
        validate_data_dir(tmp_path)


def test_run_finetune_dry_run(tmp_path: Path) -> None:
    # Create valid data dir
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text('{"messages": []}\n')
    (data_dir / "valid.jsonl").write_text('{"messages": []}\n')

    cfg = TrainingConfig(
        data_dir=str(data_dir),
        adapter_dir=str(tmp_path / "adapter"),
    )
    result = run_finetune(cfg, dry_run=True)
    assert result == Path(tmp_path / "adapter")


def test_run_finetune_missing_data(tmp_path: Path) -> None:
    cfg = TrainingConfig(
        data_dir=str(tmp_path / "nonexistent"),
        adapter_dir=str(tmp_path / "adapter"),
    )
    with pytest.raises(FileNotFoundError):
        run_finetune(cfg)

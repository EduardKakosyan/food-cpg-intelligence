"""Smoke tests for the dry-run path of scripts/remote_train_unsloth.py.

The script imports unsloth at the bottom-level _train(), so we can exercise
_load_config, _print_config_summary, and _validate_data_paths locally
without a GPU. Real ML training stays out of scope of the test suite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "remote_train_unsloth.py"


def _load_script_module():
    """Load remote_train_unsloth as a module without invoking main()."""
    spec = importlib.util.spec_from_file_location("remote_train_unsloth", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["remote_train_unsloth"] = module
    spec.loader.exec_module(module)
    return module


_module = _load_script_module()


@pytest.mark.unit
def test_load_config_round_trips_yaml(tmp_path: Path) -> None:
    cfg = {"model": "test", "lora": {"rank": 16, "alpha": 16}, "num_train_epochs": 2}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = _module._load_config(str(p))
    assert loaded["model"] == "test"
    assert loaded["lora"]["rank"] == 16
    assert loaded["num_train_epochs"] == 2


@pytest.mark.unit
def test_validate_data_paths_exits_when_train_missing(tmp_path: Path) -> None:
    cfg = {"data_dir": str(tmp_path / "missing")}
    with pytest.raises(SystemExit) as exc:
        _module._validate_data_paths(cfg)
    assert exc.value.code == 1


@pytest.mark.unit
def test_validate_data_paths_counts_lines(tmp_path: Path) -> None:
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    (data_dir / "val.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = {"data_dir": str(data_dir)}

    train_file, val_file = _module._validate_data_paths(cfg)
    assert train_file == data_dir / "train.jsonl"
    assert val_file == data_dir / "val.jsonl"


@pytest.mark.unit
def test_validate_data_paths_handles_missing_val(tmp_path: Path) -> None:
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = {"data_dir": str(data_dir)}

    _train_file, val_file = _module._validate_data_paths(cfg)
    assert val_file is None


@pytest.mark.unit
def test_print_config_summary_does_not_raise(capsys) -> None:
    cfg = {
        "model": "Qwen/Qwen3.5-9B",
        "lora": {
            "rank": 16,
            "alpha": 16,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "batch_size": 4,
        "grad_accumulation": 4,
        "num_train_epochs": 2,
        "learning_rate": 5e-5,
        "neftune_noise_alpha": 5,
    }
    _module._print_config_summary(cfg)
    captured = capsys.readouterr()
    assert "Qwen/Qwen3.5-9B" in captured.out
    assert "rank: 16" in captured.out
    assert "NEFTune alpha: 5" in captured.out
    assert "q_proj" in captured.out

"""MLX LoRA training orchestrator for Apple Silicon.

Wraps mlx_lm.lora with validation, memory checks, logging, and error handling.
Supports both programmatic invocation and subprocess fallback.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import structlog

from food_cpg_intelligence.training.config_models import TrainingConfig

logger = structlog.stdlib.get_logger(__name__)


def check_memory_budget(model_id: str) -> dict[str, object]:
    """Estimate memory requirements and check against available system memory.

    Returns a dict with estimated_gb, available_gb, and fits (bool).
    """
    # Rough estimates based on model size at 4-bit quantization
    model_size_estimates: dict[str, float] = {
        "Qwen/Qwen3.5-9B": 5.5,
        "Qwen/Qwen3.5-4B": 2.5,
        "google/gemma-3-4b-it": 2.5,
    }

    base_gb = model_size_estimates.get(model_id, 6.0)
    # LoRA adapters + optimizer + activations + overhead
    overhead_gb = 8.0
    estimated_gb = base_gb + overhead_gb

    # Get available memory on macOS
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        total_gb = round(page_size * total_pages / (1024**3), 1)
    except (ValueError, OSError):
        total_gb = 48.0  # assume M4 Pro default

    fits = estimated_gb < total_gb * 0.85  # leave 15% headroom

    return {
        "model": model_id,
        "estimated_gb": round(estimated_gb, 1),
        "available_gb": total_gb,
        "fits": fits,
    }


def validate_data_dir(data_dir: Path) -> None:
    """Validate that the MLX training data directory has required files."""
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation data not found: {valid_path}")

    # Check files are non-empty
    train_lines = sum(1 for _ in train_path.open())
    valid_lines = sum(1 for _ in valid_path.open())

    if train_lines == 0:
        raise ValueError(f"Training file is empty: {train_path}")
    if valid_lines == 0:
        raise ValueError(f"Validation file is empty: {valid_path}")

    logger.info("data_validated", train=train_lines, valid=valid_lines)


def run_finetune(config: TrainingConfig, *, dry_run: bool = False) -> Path:
    """Run MLX LoRA fine-tuning.

    Args:
        config: Training configuration.
        dry_run: If True, validate config and data without training.

    Returns:
        Path to the adapter output directory.
    """
    data_dir = Path(config.data_dir)
    adapter_dir = Path(config.adapter_dir)

    # Validate data
    validate_data_dir(data_dir)

    # Check memory
    mem = check_memory_budget(config.model)
    logger.info("memory_check", **mem)
    if not mem["fits"]:
        logger.warning(
            "memory_tight",
            estimated=mem["estimated_gb"],
            available=mem["available_gb"],
            suggestion="Reduce batch_size to 1 or enable grad_checkpoint",
        )

    if dry_run:
        logger.info("dry_run_complete", config=config.to_mlx_dict())
        return adapter_dir

    # Write mlx_lm config to temp file
    mlx_yaml = config.to_mlx_yaml()
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="mlx_train_"
    ) as f:
        f.write(mlx_yaml)
        config_path = f.name

    logger.info("training_start", config_path=config_path, model=config.model)

    try:
        # Try programmatic invocation first
        _run_mlx_lora_programmatic(config_path)
    except ImportError:
        logger.warning("mlx_lm_import_failed", fallback="subprocess")
        _run_mlx_lora_subprocess(config_path)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    logger.info("training_complete", adapter_dir=str(adapter_dir))
    return adapter_dir


def _run_mlx_lora_programmatic(config_path: str) -> None:
    """Invoke mlx_lm.lora training programmatically via run()."""
    from mlx_lm import lora

    # Build args namespace from config YAML, same as CLI --config does
    parser = lora.build_parser()  # type: ignore[no-untyped-call]
    args = parser.parse_args(["--config", config_path])
    lora.run(args)


def _run_mlx_lora_subprocess(config_path: str) -> None:
    """Invoke mlx_lm.lora training via subprocess (fallback)."""
    cmd = ["python", "-m", "mlx_lm.lora", "--config", config_path]
    logger.info("subprocess_start", cmd=" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        logger.error("training_failed", stderr=result.stderr[:500])
        raise RuntimeError(f"mlx_lm.lora failed (exit {result.returncode}): {result.stderr[:500]}")

    if result.stdout:
        logger.info("training_stdout", output=result.stdout[-500:])

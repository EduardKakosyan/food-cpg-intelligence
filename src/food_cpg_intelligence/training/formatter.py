"""Format training data for Unsloth QLoRA fine-tuning.

Supports Qwen 3.5 ChatML format and outputs HuggingFace datasets Arrow files.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import structlog

from food_cpg_intelligence.training.models import TrainingTriple

logger = structlog.stdlib.get_logger(__name__)

SYSTEM_PROMPT = (
    "You are Ask Peter -- an AI assistant that channels Peter Chapman's decades "
    "of Canadian CPG expertise. You give practical, specific advice to food and "
    "beverage SMEs on retailer meetings, pricing strategy, trade shows, "
    "promotion planning, and category management. Ground your answers in real "
    "industry knowledge and be direct."
)


def format_chatml(triple: TrainingTriple) -> dict[str, str]:
    """Format a training triple as a Qwen 3.5 ChatML conversation.

    Qwen 3.5 uses the ChatML template:
        <|im_start|>system\\n{system}<|im_end|>
        <|im_start|>user\\n{instruction}<|im_end|>
        <|im_start|>assistant\\n{response}<|im_end|>
    """
    user_content = triple.instruction
    if triple.context:
        user_content = f"{triple.instruction}\n\nContext:\n{triple.context}"

    conversation = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{triple.response}<|im_end|>"
    )
    return {"text": conversation}


def format_dataset(
    triples: list[TrainingTriple],
    *,
    seed: int = 42,
    train_fraction: float = 0.9,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Format triples into train/val splits of ChatML conversations.

    Args:
        triples: List of training triples to format.
        seed: Random seed for reproducible splitting.
        train_fraction: Fraction of data for training (rest is validation).

    Returns:
        Tuple of (train_examples, val_examples), each a list of {"text": "..."} dicts.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    formatted = [format_chatml(t) for t in triples]

    rng = random.Random(seed)
    rng.shuffle(formatted)

    split_idx = int(len(formatted) * train_fraction)
    train = formatted[:split_idx]
    val = formatted[split_idx:]

    logger.info("dataset_split", train=len(train), val=len(val), total=len(formatted))
    return train, val


def save_jsonl_dataset(
    examples: list[dict[str, str]],
    path: Path,
) -> Path:
    """Save formatted examples as JSONL (one {"text": "..."} per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("saved_dataset", path=str(path), count=len(examples))
    return path


def save_formatted_dataset(
    triples: list[TrainingTriple],
    output_dir: Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.9,
) -> tuple[Path, Path]:
    """Format, split, and save training data as JSONL files.

    Args:
        triples: Filtered training triples.
        output_dir: Directory to save train.jsonl and val.jsonl.
        seed: Random seed for split.
        train_fraction: Train/val split ratio.

    Returns:
        Tuple of (train_path, val_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train, val = format_dataset(triples, seed=seed, train_fraction=train_fraction)

    train_path = save_jsonl_dataset(train, output_dir / "train.jsonl")
    val_path = save_jsonl_dataset(val, output_dir / "val.jsonl")

    # Save metadata
    meta = {
        "system_prompt": SYSTEM_PROMPT,
        "format": "chatml",
        "model_target": "Qwen/Qwen3.5-9B",
        "train_count": len(train),
        "val_count": len(val),
        "total": len(train) + len(val),
        "seed": seed,
        "train_fraction": train_fraction,
    }
    meta_path = output_dir / "dataset_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return train_path, val_path

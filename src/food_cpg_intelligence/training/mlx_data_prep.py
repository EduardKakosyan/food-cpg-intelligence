"""Convert training data to mlx-lm chat messages format.

mlx-lm expects JSONL with {"messages": [{"role": "...", "content": "..."}]} format.
This module converts from TrainingTriple objects or existing ChatML text JSONL.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from food_cpg_intelligence.training.formatter import SYSTEM_PROMPT
from food_cpg_intelligence.training.models import TrainingTriple

logger = structlog.stdlib.get_logger(__name__)


def triple_to_chat_messages(triple: TrainingTriple) -> dict[str, list[dict[str, str]]]:
    """Convert a TrainingTriple to mlx-lm chat messages format.

    Returns {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
    """
    user_content = triple.instruction
    if triple.context:
        user_content = f"{triple.instruction}\n\nContext:\n{triple.context}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": triple.response},
        ]
    }


def load_triples_from_jsonl(path: Path) -> list[TrainingTriple]:
    """Load TrainingTriple objects from a JSONL file."""
    triples: list[TrainingTriple] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                try:
                    triples.append(TrainingTriple.model_validate_json(stripped))
                except Exception as exc:
                    logger.warning("triple_parse_skip", error=str(exc))
    return triples


def save_mlx_dataset(
    triples: list[TrainingTriple],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Convert triples to mlx-lm format and save as train.jsonl + valid.jsonl.

    Note: Uses the already-split data from the formatter output.
    This function takes a pre-split list and writes it as a single file.
    Call separately for train and val splits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "data.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for triple in triples:
            msg = triple_to_chat_messages(triple)
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    logger.info("saved_mlx_dataset", path=str(path), count=len(triples))
    return path, output_dir


def prepare_mlx_data(
    filtered_pairs_path: Path,
    output_dir: Path,
    *,
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Full pipeline: load filtered triples → convert → split → save as mlx-lm format.

    Creates output_dir/train.jsonl and output_dir/valid.jsonl in chat messages format.
    """
    import random

    triples = load_triples_from_jsonl(filtered_pairs_path)
    if not triples:
        raise ValueError(f"No valid triples found in {filtered_pairs_path}")

    messages = [triple_to_chat_messages(t) for t in triples]

    rng = random.Random(seed)
    rng.shuffle(messages)

    split_idx = int(len(messages) * train_fraction)
    train_messages = messages[:split_idx]
    val_messages = messages[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as f:
        for msg in train_messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    val_path = output_dir / "valid.jsonl"
    with val_path.open("w", encoding="utf-8") as f:
        for msg in val_messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    logger.info(
        "prepared_mlx_data",
        train=len(train_messages),
        val=len(val_messages),
        output_dir=str(output_dir),
    )
    return train_path, val_path

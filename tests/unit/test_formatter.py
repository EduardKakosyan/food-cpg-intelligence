"""Tests for training data formatter."""

import json
from pathlib import Path

import pytest

from food_cpg_intelligence.training.formatter import (
    SYSTEM_PROMPT,
    format_chatml,
    format_dataset,
    save_formatted_dataset,
    save_jsonl_dataset,
)
from food_cpg_intelligence.training.models import TrainingTriple


def _make_triple() -> TrainingTriple:
    return TrainingTriple(
        triple_id="t1",
        instruction="How do I approach Loblaw?",
        context="Focus on data and category performance.",
        response="You need to bring strong data to the meeting.",
    )


def test_format_chatml_structure() -> None:
    t = _make_triple()
    result = format_chatml(t)
    assert "text" in result
    text = result["text"]
    assert "<|im_start|>system" in text
    assert SYSTEM_PROMPT in text
    assert "<|im_start|>user" in text
    assert "<|im_start|>assistant" in text
    assert "<|im_end|>" in text
    assert t.instruction in text
    assert t.response in text


def test_format_chatml_includes_context() -> None:
    t = _make_triple()
    result = format_chatml(t)
    assert "Context:" in result["text"]
    assert t.context in result["text"]


def test_format_chatml_no_context() -> None:
    t = TrainingTriple(
        triple_id="t1",
        instruction="General question?",
        context="",
        response="General answer.",
    )
    result = format_chatml(t)
    assert "Context:" not in result["text"]


def test_format_dataset_split() -> None:
    triples = [
        TrainingTriple(
            triple_id=f"t{i}",
            instruction=f"Q{i}?",
            context=f"C{i}",
            response=f"R{i}",
        )
        for i in range(100)
    ]
    train, val = format_dataset(triples, seed=42, train_fraction=0.9)
    assert len(train) == 90
    assert len(val) == 10
    assert all("text" in ex for ex in train)
    assert all("text" in ex for ex in val)


def test_format_dataset_deterministic() -> None:
    triples = [
        TrainingTriple(triple_id=f"t{i}", instruction=f"Q{i}", context="", response=f"R{i}")
        for i in range(20)
    ]
    train1, val1 = format_dataset(triples, seed=42)
    train2, val2 = format_dataset(triples, seed=42)
    assert train1 == train2
    assert val1 == val2


def test_format_dataset_rejects_invalid_train_fraction() -> None:
    triples = [TrainingTriple(triple_id="t0", instruction="Q?", context="C", response="R")]
    with pytest.raises(ValueError, match="train_fraction"):
        format_dataset(triples, train_fraction=0.0)
    with pytest.raises(ValueError, match="train_fraction"):
        format_dataset(triples, train_fraction=1.0)
    with pytest.raises(ValueError, match="train_fraction"):
        format_dataset(triples, train_fraction=1.5)


def test_save_jsonl_dataset(tmp_path: Path) -> None:
    examples = [{"text": "example 1"}, {"text": "example 2"}]
    path = tmp_path / "test.jsonl"
    save_jsonl_dataset(examples, path)
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "example 1"


def test_save_formatted_dataset_creates_files(tmp_path: Path) -> None:
    triples = [
        TrainingTriple(triple_id=f"t{i}", instruction=f"Q{i}", context="C", response=f"R{i}")
        for i in range(20)
    ]
    train_path, val_path = save_formatted_dataset(triples, tmp_path, seed=42)
    assert train_path.exists()
    assert val_path.exists()
    assert (tmp_path / "dataset_meta.json").exists()

    meta = json.loads((tmp_path / "dataset_meta.json").read_text())
    assert meta["format"] == "chatml"
    assert meta["train_count"] == 18  # 90% of 20
    assert meta["val_count"] == 2

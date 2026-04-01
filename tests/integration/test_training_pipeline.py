"""Integration test: raw triples -> quality filter -> format pipeline."""

import json
from pathlib import Path
from typing import Literal

from food_cpg_intelligence.evaluation.models import Category, GoldStandardItem
from food_cpg_intelligence.training.formatter import save_formatted_dataset
from food_cpg_intelligence.training.models import TrainingTriple, compute_stats
from food_cpg_intelligence.training.quality_filter import run_quality_pipeline


def _make_raw_triples(n: int = 50) -> list[TrainingTriple]:
    """Generate synthetic training triples for testing."""
    categories = [
        "general_cpg",
        "retailer_strategy",
        "pricing_promotion",
        "trade_shows",
        "product_launch",
    ]
    types: list[Literal["factual", "analytical", "advice", "scenario"]] = [
        "factual",
        "analytical",
        "advice",
        "scenario",
    ]
    return [
        TrainingTriple(
            triple_id=f"train-test-{i:03d}",
            instruction=f"Question {i}: How does this apply to {categories[i % 5]}?",
            context=f"In newsletter {100 + i}, Peter discusses {categories[i % 5]} strategies.",
            response=f"Based on Peter's advice in the newsletter, you should focus on {categories[i % 5]}. "
            f"This is practical advice number {i} that covers the key points about "
            f"Canadian food industry practices and how they relate to your business. "
            f"Consider the retail landscape and your specific category position.",
            source_doc_id=f"newsletter-{100 + i}",
            category=categories[i % 5],
            question_type=types[i % 4],
        )
        for i in range(n)
    ]


def _make_gold_items() -> list[GoldStandardItem]:
    return [
        GoldStandardItem(
            question_id="g1",
            question="Question 0: How does this apply to general_cpg?",
            reference_answer="Answer",
            category=Category.GENERAL_CPG,
        ),
    ]


def test_full_training_pipeline(tmp_path: Path) -> None:
    """End-to-end: raw triples -> filter -> format -> verify."""
    raw = _make_raw_triples(50)
    gold = _make_gold_items()

    # Quality filter
    filtered, stats = run_quality_pipeline(raw, gold)
    assert stats["input"] == 50
    assert stats["contamination_removed"] >= 1
    assert len(filtered) < 50

    # Compute stats
    ds_stats = compute_stats(filtered, contamination_removed=stats["contamination_removed"])
    assert ds_stats.total == len(filtered)
    assert ds_stats.contamination_removed >= 1

    # Format for Unsloth
    train_path, val_path = save_formatted_dataset(filtered, tmp_path / "formatted")
    assert train_path.exists()
    assert val_path.exists()

    # Verify train JSONL content
    with train_path.open() as f:
        first_line = json.loads(f.readline())
    assert "text" in first_line
    assert "<|im_start|>system" in first_line["text"]
    assert "<|im_start|>user" in first_line["text"]
    assert "<|im_start|>assistant" in first_line["text"]

    # Verify metadata
    meta = json.loads((tmp_path / "formatted" / "dataset_meta.json").read_text())
    assert meta["format"] == "chatml"
    assert meta["train_count"] + meta["val_count"] == len(filtered)

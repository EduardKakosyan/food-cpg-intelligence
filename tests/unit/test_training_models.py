"""Tests for training data models."""

import pytest
from pydantic import ValidationError

from food_cpg_intelligence.training.models import TrainingTriple, compute_stats


def _make_triples(n: int = 5) -> list[TrainingTriple]:
    return [
        TrainingTriple(
            triple_id=f"train-0-{i:03d}",
            instruction=f"Question {i} about CPG?",
            context=f"Context passage {i} from a newsletter.",
            response=f"Peter says: here is advice number {i} about the food industry.",
            source_doc_id=f"newsletter-{100 + i}",
            category="general_cpg" if i % 2 == 0 else "retailer_strategy",
            question_type="factual" if i % 2 == 0 else "advice",
        )
        for i in range(n)
    ]


def test_training_triple_frozen() -> None:
    t = TrainingTriple(
        triple_id="t1",
        instruction="Q?",
        context="C",
        response="R",
    )
    with pytest.raises(ValidationError):
        t.instruction = "mutated"  # type: ignore[misc]


def test_training_triple_defaults() -> None:
    t = TrainingTriple(
        triple_id="t1",
        instruction="Q?",
        context="C",
        response="R",
    )
    assert t.question_type == "factual"
    assert t.quality_score is None
    assert t.category == ""


def test_compute_stats() -> None:
    triples = _make_triples(10)
    stats = compute_stats(triples)
    assert stats.total == 10
    assert stats.by_category["general_cpg"] == 5
    assert stats.by_category["retailer_strategy"] == 5
    assert stats.by_question_type["factual"] == 5
    assert stats.avg_instruction_length > 0
    assert stats.avg_response_length > 0


def test_compute_stats_empty() -> None:
    stats = compute_stats([])
    assert stats.total == 0


def test_compute_stats_with_removal_counts() -> None:
    stats = compute_stats(_make_triples(3), dedup_removed=2, contamination_removed=1)
    assert stats.dedup_removed == 2
    assert stats.contamination_removed == 1

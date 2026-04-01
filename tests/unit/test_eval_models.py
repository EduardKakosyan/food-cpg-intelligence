"""Tests for evaluation data models."""

import pytest
from pydantic import ValidationError

from food_cpg_intelligence.evaluation.models import (
    BlindBatch,
    BlindItem,
    Category,
    EvalReport,
    EvalResponse,
    GoldStandardItem,
    GoldStandardSet,
    JudgeScore,
)


def test_category_enum_values() -> None:
    assert Category.RETAILER_STRATEGY.value == "retailer_strategy"
    assert Category.UNANSWERABLE.value == "unanswerable"
    assert len(Category) == 6


def test_gold_standard_item_frozen() -> None:
    item = GoldStandardItem(
        question_id="test-001",
        question="What is X?",
        reference_answer="X is Y.",
        category=Category.GENERAL_CPG,
    )
    assert item.difficulty == "medium"
    assert item.verified is False
    with pytest.raises(ValidationError):
        item.question = "mutated"  # type: ignore[misc]


def test_gold_standard_set_category_counts() -> None:
    items = (
        GoldStandardItem(
            question_id="a",
            question="Q1",
            reference_answer="A1",
            category=Category.RETAILER_STRATEGY,
        ),
        GoldStandardItem(
            question_id="b",
            question="Q2",
            reference_answer="A2",
            category=Category.RETAILER_STRATEGY,
        ),
        GoldStandardItem(
            question_id="c",
            question="Q3",
            reference_answer="A3",
            category=Category.PRICING_PROMOTION,
        ),
    )
    gs = GoldStandardSet(items=items)
    assert gs.category_counts == {"retailer_strategy": 2, "pricing_promotion": 1}


def test_gold_standard_set_difficulty_counts() -> None:
    items = (
        GoldStandardItem(
            question_id="a",
            question="Q",
            reference_answer="A",
            category=Category.GENERAL_CPG,
            difficulty="easy",
        ),
        GoldStandardItem(
            question_id="b",
            question="Q",
            reference_answer="A",
            category=Category.GENERAL_CPG,
            difficulty="hard",
        ),
    )
    gs = GoldStandardSet(items=items)
    assert gs.difficulty_counts == {"easy": 1, "hard": 1}


def test_eval_response_with_context() -> None:
    resp = EvalResponse(
        question_id="q1",
        system="rag",
        response="answer",
        context_used=("chunk-1", "chunk-2"),
    )
    assert resp.context_used is not None
    assert len(resp.context_used) == 2


def test_eval_response_without_context() -> None:
    resp = EvalResponse(question_id="q1", system="finetune", response="answer")
    assert resp.context_used is None


def test_judge_score_dimension_scores() -> None:
    score = JudgeScore(
        question_id="q1",
        system="rag",
        accuracy=4.0,
        grounding=3.5,
        completeness=4.0,
        voice_fidelity=3.0,
        hallucination_resistance=5.0,
        overall=4.0,
        reasoning="Good response.",
    )
    dims = score.dimension_scores
    assert dims["accuracy"] == 4.0
    assert dims["hallucination_resistance"] == 5.0
    assert len(dims) == 5


def test_blind_batch_creation() -> None:
    batch = BlindBatch(
        items=(
            BlindItem(
                blind_id="b1",
                question_id="q1",
                question="Q?",
                reference_answer="A",
                response="resp",
                category=Category.GENERAL_CPG,
            ),
        ),
        id_to_system={"b1": "rag"},
    )
    assert len(batch.items) == 1
    assert batch.id_to_system["b1"] == "rag"


def test_eval_report_defaults() -> None:
    report = EvalReport()
    assert report.systems == ()
    assert report.scores_by_system == {}

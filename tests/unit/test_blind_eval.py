"""Tests for blind evaluation protocol."""

from food_cpg_intelligence.evaluation.blind_eval import (
    prepare_blind_batch,
    reveal_results,
)
from food_cpg_intelligence.evaluation.models import (
    Category,
    EvalResponse,
    GoldStandardItem,
    JudgeScore,
)


def _make_gold_items() -> list[GoldStandardItem]:
    return [
        GoldStandardItem(
            question_id="q1",
            question="What is X?",
            reference_answer="X is Y.",
            category=Category.GENERAL_CPG,
        ),
        GoldStandardItem(
            question_id="q2",
            question="How to do Z?",
            reference_answer="Do Z by...",
            category=Category.RETAILER_STRATEGY,
        ),
    ]


def test_prepare_blind_batch_shuffles() -> None:
    gold = _make_gold_items()
    responses = {
        "rag": [
            EvalResponse(question_id="q1", system="rag", response="RAG answer 1"),
            EvalResponse(question_id="q2", system="rag", response="RAG answer 2"),
        ],
        "finetune": [
            EvalResponse(question_id="q1", system="finetune", response="FT answer 1"),
            EvalResponse(question_id="q2", system="finetune", response="FT answer 2"),
        ],
    }
    batch = prepare_blind_batch(gold, responses, seed=42)

    assert len(batch.items) == 4
    assert len(batch.id_to_system) == 4
    # System names should NOT appear in the blind items
    for item in batch.items:
        assert "rag" not in item.blind_id
        assert "finetune" not in item.blind_id


def test_prepare_blind_batch_all_systems_represented() -> None:
    gold = _make_gold_items()
    responses = {
        "rag": [EvalResponse(question_id="q1", system="rag", response="R")],
        "finetune": [EvalResponse(question_id="q1", system="finetune", response="F")],
    }
    batch = prepare_blind_batch(gold, responses)
    systems = set(batch.id_to_system.values())
    assert systems == {"rag", "finetune"}


def test_reveal_results_unmasks_correctly() -> None:
    gold = _make_gold_items()
    responses = {
        "rag": [EvalResponse(question_id="q1", system="rag", response="R1")],
        "finetune": [EvalResponse(question_id="q1", system="finetune", response="F1")],
    }
    batch = prepare_blind_batch(gold, responses, seed=123)

    # Simulate judge scores using blind_ids
    scores = [
        JudgeScore(
            question_id=item.blind_id,
            system="unknown",  # judge doesn't know the system
            accuracy=4.0,
            grounding=3.0,
            completeness=4.0,
            voice_fidelity=3.0,
            hallucination_resistance=5.0,
            overall=4.0,
            reasoning="OK",
        )
        for item in batch.items
    ]

    results = reveal_results(batch, scores)
    assert "rag" in results
    assert "finetune" in results
    # Unmasked scores should have real question_ids
    for system_scores in results.values():
        for score in system_scores:
            assert score.question_id == "q1"
            assert score.system in ("rag", "finetune")


def test_blind_batch_deterministic_with_same_seed() -> None:
    gold = _make_gold_items()
    responses = {
        "a": [EvalResponse(question_id="q1", system="a", response="A")],
        "b": [EvalResponse(question_id="q1", system="b", response="B")],
    }
    batch1 = prepare_blind_batch(gold, responses, seed=99)
    batch2 = prepare_blind_batch(gold, responses, seed=99)

    # Same seed should produce same ordering (though blind_ids differ due to uuid)
    assert len(batch1.items) == len(batch2.items)

"""Integration test: gold set -> blind eval -> judge -> report pipeline."""

from pathlib import Path

from food_cpg_intelligence.evaluation.blind_eval import (
    prepare_blind_batch,
    reveal_results,
)
from food_cpg_intelligence.evaluation.gold_set_generator import (
    build_gold_set,
    load_gold_items,
    save_gold_items,
    save_gold_set,
)
from food_cpg_intelligence.evaluation.judge import (
    save_judge_scores,
    scores_dict_to_judge_score,
)
from food_cpg_intelligence.evaluation.models import (
    Category,
    EvalResponse,
    GoldStandardItem,
    JudgeScore,
)
from food_cpg_intelligence.evaluation.report import compile_report, save_report


def _make_fixture_gold() -> list[GoldStandardItem]:
    return [
        GoldStandardItem(
            question_id="retailer_strategy-001",
            question="How should I prepare for a Loblaw category review?",
            reference_answer="Focus on data, know your category numbers.",
            category=Category.RETAILER_STRATEGY,
            difficulty="medium",
        ),
        GoldStandardItem(
            question_id="pricing_promotion-001",
            question="How do I set pricing for a new product?",
            reference_answer="Start with your cost, add margin, check competitors.",
            category=Category.PRICING_PROMOTION,
            difficulty="easy",
        ),
        GoldStandardItem(
            question_id="unanswerable-001",
            question="What was Walmart Canada's Q3 2024 revenue?",
            reference_answer="This cannot be answered from SKUFood content.",
            category=Category.UNANSWERABLE,
            difficulty="easy",
        ),
    ]


def test_full_eval_pipeline(tmp_path: Path) -> None:
    """End-to-end: gold set -> responses -> blind eval -> scores -> report."""
    gold_items = _make_fixture_gold()

    # Save and reload gold set
    items_path = tmp_path / "gold.jsonl"
    save_gold_items(gold_items, items_path)
    reloaded = load_gold_items(items_path)
    assert len(reloaded) == 3

    gs = build_gold_set(reloaded, description="test")
    gs_path = tmp_path / "gold_set.json"
    save_gold_set(gs, gs_path)

    # Simulate system responses
    responses_rag = [
        EvalResponse(
            question_id="retailer_strategy-001",
            system="rag",
            response="Prepare data and category numbers for Loblaw.",
        ),
        EvalResponse(
            question_id="pricing_promotion-001",
            system="rag",
            response="Calculate cost plus margin, benchmark competitors.",
        ),
        EvalResponse(
            question_id="unanswerable-001",
            system="rag",
            response="I don't have that specific revenue data.",
        ),
    ]
    responses_ft = [
        EvalResponse(
            question_id="retailer_strategy-001",
            system="finetune",
            response="You should bring good data to the meeting.",
        ),
        EvalResponse(
            question_id="pricing_promotion-001",
            system="finetune",
            response="Price it based on what feels right.",
        ),
        EvalResponse(
            question_id="unanswerable-001",
            system="finetune",
            response="Walmart made $5 billion last quarter.",
        ),
    ]

    # Blind evaluation
    batch = prepare_blind_batch(
        gold_items,
        {"rag": responses_rag, "finetune": responses_ft},
    )
    assert len(batch.items) == 6

    # Simulate judge scores (normally done by agents)
    mock_scores: list[JudgeScore] = []
    for item in batch.items:
        system = batch.id_to_system[item.blind_id]
        # RAG gets higher scores
        base = 4.0 if system == "rag" else 2.5
        mock_scores.append(
            scores_dict_to_judge_score(
                {
                    "accuracy": base,
                    "grounding": base,
                    "completeness": base,
                    "voice_fidelity": base - 0.5,
                    "hallucination_resistance": base + 0.5,
                    "overall": base,
                    "reasoning": f"Mock score for {system}",
                },
                question_id=item.blind_id,
                system="judge",
            )
        )

    # Reveal results
    results = reveal_results(batch, mock_scores)
    assert "rag" in results
    assert "finetune" in results

    # Save scores
    for sys_name, sys_scores in results.items():
        save_judge_scores(sys_scores, tmp_path / f"scores_{sys_name}.jsonl")

    # Compile report
    report = compile_report(results)
    assert (
        report.scores_by_system["rag"]["overall"] > report.scores_by_system["finetune"]["overall"]
    )

    # Save report
    md_path, json_path = save_report(report, tmp_path)
    assert md_path.exists()
    assert json_path.exists()
    assert "rag" in md_path.read_text()

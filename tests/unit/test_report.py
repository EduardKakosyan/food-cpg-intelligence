"""Tests for evaluation report generator."""

from food_cpg_intelligence.evaluation.models import JudgeScore
from food_cpg_intelligence.evaluation.report import (
    _bootstrap_ci,
    _mean,
    compile_report,
    render_markdown,
)


def _make_scores(system: str, n: int = 5) -> list[JudgeScore]:
    return [
        JudgeScore(
            question_id=f"general_cpg-{i:03d}",
            system=system,
            accuracy=4.0 + (i % 2) * 0.5,
            grounding=3.5,
            completeness=4.0,
            voice_fidelity=3.0 + (i % 3) * 0.5,
            hallucination_resistance=4.5,
            overall=4.0,
            reasoning=f"Score {i}",
        )
        for i in range(n)
    ]


def test_mean_empty() -> None:
    assert _mean([]) == 0.0


def test_mean_values() -> None:
    assert _mean([2.0, 4.0]) == 3.0


def test_bootstrap_ci_single_value() -> None:
    low, high = _bootstrap_ci([3.0])
    assert low == 3.0
    assert high == 3.0


def test_bootstrap_ci_range() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    low, high = _bootstrap_ci(values)
    assert low < high
    assert low >= 1.0
    assert high <= 5.0


def test_compile_report_single_system() -> None:
    scores = {"rag": _make_scores("rag", 10)}
    report = compile_report(scores)
    assert "rag" in report.systems
    assert "accuracy" in report.scores_by_system["rag"]
    assert report.scores_by_system["rag"]["overall"] > 0


def test_compile_report_two_systems_head_to_head() -> None:
    scores_a = _make_scores("rag", 5)
    scores_b = [
        JudgeScore(
            question_id=s.question_id,
            system="finetune",
            accuracy=s.accuracy - 1,
            grounding=s.grounding - 1,
            completeness=s.completeness,
            voice_fidelity=s.voice_fidelity,
            hallucination_resistance=s.hallucination_resistance,
            overall=s.overall - 1,
            reasoning="Lower",
        )
        for s in scores_a
    ]
    report = compile_report({"rag": scores_a, "finetune": scores_b})
    assert "rag" in report.head_to_head
    assert "finetune" in report.head_to_head
    assert report.head_to_head["rag"]["wins"] > 0


def test_compile_report_per_category() -> None:
    scores = {"rag": _make_scores("rag", 5)}
    report = compile_report(scores)
    assert "general_cpg" in report.scores_by_category


def test_render_markdown_contains_tables() -> None:
    scores = {"rag": _make_scores("rag", 5)}
    report = compile_report(scores)
    md = render_markdown(report)
    assert "# Evaluation Report" in md
    assert "| rag |" in md
    assert "Accuracy" in md

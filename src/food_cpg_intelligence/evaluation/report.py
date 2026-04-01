"""Evaluation report generator — Markdown with per-system, per-category breakdowns."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

import structlog

from food_cpg_intelligence.evaluation.models import (
    EvalReport,
    JudgeScore,
)

logger = structlog.stdlib.get_logger(__name__)

DIMENSIONS = (
    "accuracy",
    "grounding",
    "completeness",
    "voice_fidelity",
    "hallucination_resistance",
    "overall",
)


def _mean(values: list[float]) -> float:
    """Safe mean that returns 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _bootstrap_ci(
    values: list[float],
    *,
    n_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if len(values) < 2:
        m = _mean(values)
        return (m, m)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_samples):
        sample = rng.choices(values, k=len(values))
        means.append(_mean(sample))

    means.sort()
    lower_idx = int((1 - confidence) / 2 * n_samples)
    upper_idx = int((1 + confidence) / 2 * n_samples) - 1
    return (round(means[lower_idx], 3), round(means[upper_idx], 3))


def _compute_system_averages(
    scores_by_system: dict[str, list[JudgeScore]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]]]:
    """Compute per-system dimension averages and bootstrap CIs."""
    sys_avgs: dict[str, dict[str, float]] = {}
    sys_cis: dict[str, dict[str, tuple[float, float]]] = {}
    for sys_name, scores in scores_by_system.items():
        dim_values: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
        for s in scores:
            for d in DIMENSIONS:
                dim_values[d].append(getattr(s, d))
        sys_avgs[sys_name] = {d: round(_mean(v), 3) for d, v in dim_values.items()}
        sys_cis[sys_name] = {d: _bootstrap_ci(v) for d, v in dim_values.items()}
    return sys_avgs, sys_cis


def _compute_category_scores(
    scores_by_system: dict[str, list[JudgeScore]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute per-category, per-system dimension averages."""
    cat_scores: dict[str, dict[str, dict[str, float]]] = {}
    for sys_name, scores in scores_by_system.items():
        by_cat: dict[str, list[JudgeScore]] = {}
        for s in scores:
            cat = s.question_id.rsplit("-", 1)[0] if "-" in s.question_id else "unknown"
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(s)

        for cat, cat_items in by_cat.items():
            if cat not in cat_scores:
                cat_scores[cat] = {}
            dim_vals = {d: [getattr(s, d) for s in cat_items] for d in DIMENSIONS}
            cat_scores[cat][sys_name] = {d: round(_mean(v), 3) for d, v in dim_vals.items()}
    return cat_scores


def _compute_head_to_head(
    scores_by_system: dict[str, list[JudgeScore]],
    systems: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """Compute head-to-head wins/losses/ties for two-system comparisons."""
    if len(systems) != 2:
        return {}
    sys_a, sys_b = systems
    scores_a = {s.question_id: s.overall for s in scores_by_system[sys_a]}
    scores_b = {s.question_id: s.overall for s in scores_by_system[sys_b]}
    wins_a = wins_b = ties = 0
    for qid, score_a in scores_a.items():
        if qid in scores_b:
            if score_a > scores_b[qid]:
                wins_a += 1
            elif scores_b[qid] > score_a:
                wins_b += 1
            else:
                ties += 1
    return {
        sys_a: {"wins": wins_a, "losses": wins_b, "ties": ties},
        sys_b: {"wins": wins_b, "losses": wins_a, "ties": ties},
    }


def compile_report(
    scores_by_system: dict[str, list[JudgeScore]],
) -> EvalReport:
    """Compile judge scores into an EvalReport with aggregated metrics."""
    systems = tuple(sorted(scores_by_system.keys()))
    sys_avgs, sys_cis = _compute_system_averages(scores_by_system)
    cat_scores = _compute_category_scores(scores_by_system)
    h2h = _compute_head_to_head(scores_by_system, systems)

    return EvalReport(
        systems=systems,
        scores_by_system=sys_avgs,
        scores_by_category=cat_scores,
        confidence_intervals=sys_cis,
        head_to_head=h2h,
        metadata={
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "total_scores": sum(len(s) for s in scores_by_system.values()),
        },
    )


def render_markdown(report: EvalReport) -> str:
    """Render an EvalReport as a Markdown document."""
    lines: list[str] = [
        "# Evaluation Report",
        "",
        f"Generated: {report.metadata.get('generated_at', 'N/A')}",
        f"Total scored items: {report.metadata.get('total_scores', 0)}",
        "",
        "## Overall Scores by System",
        "",
        "| System | Accuracy | Grounding | Completeness | Voice | Halluc. Resist. | Overall |",
        "|--------|----------|-----------|--------------|-------|-----------------|---------|",
    ]

    for sys_name in report.systems:
        s = report.scores_by_system.get(sys_name, {})
        lines.append(
            f"| {sys_name} | {s.get('accuracy', 0):.2f} | {s.get('grounding', 0):.2f} "
            f"| {s.get('completeness', 0):.2f} | {s.get('voice_fidelity', 0):.2f} "
            f"| {s.get('hallucination_resistance', 0):.2f} | {s.get('overall', 0):.2f} |"
        )

    # Confidence intervals
    if report.confidence_intervals:
        lines.extend(["", "## Confidence Intervals (95%)", ""])
        for sys_name in report.systems:
            cis = report.confidence_intervals.get(sys_name, {})
            overall_ci = cis.get("overall", (0, 0))
            lines.append(f"- **{sys_name}** overall: {overall_ci[0]:.2f} - {overall_ci[1]:.2f}")

    # Per-category breakdown
    if report.scores_by_category:
        lines.extend(["", "## Scores by Category", ""])
        for cat in sorted(report.scores_by_category.keys()):
            lines.append(f"### {cat}")
            lines.append("| System | Accuracy | Grounding | Completeness | Voice | Overall |")
            lines.append("|--------|----------|-----------|--------------|-------|---------|")
            for sys_name in report.systems:
                s = report.scores_by_category[cat].get(sys_name, {})
                lines.append(
                    f"| {sys_name} | {s.get('accuracy', 0):.2f} "
                    f"| {s.get('grounding', 0):.2f} | {s.get('completeness', 0):.2f} "
                    f"| {s.get('voice_fidelity', 0):.2f} | {s.get('overall', 0):.2f} |"
                )
            lines.append("")

    # Head-to-head
    if report.head_to_head:
        lines.extend(["## Head-to-Head", ""])
        for sys_name, h2h in report.head_to_head.items():
            lines.append(
                f"- **{sys_name}**: {h2h.get('wins', 0)} wins, "
                f"{h2h.get('losses', 0)} losses, {h2h.get('ties', 0)} ties"
            )

    lines.append("")
    return "\n".join(lines)


def save_report(report: EvalReport, output_dir: Path) -> tuple[Path, Path]:
    """Save report as both Markdown and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "eval_report.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    json_path = output_dir / "eval_report.json"
    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, default=str),
        encoding="utf-8",
    )

    logger.info("saved_report", md=str(md_path), json=str(json_path))
    return md_path, json_path

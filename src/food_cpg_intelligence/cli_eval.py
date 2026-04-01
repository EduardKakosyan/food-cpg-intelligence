"""CLI subcommands for evaluation."""

from __future__ import annotations

from pathlib import Path

import typer

from food_cpg_intelligence.config import settings
from food_cpg_intelligence.evaluation.gold_set_generator import (
    build_gold_set,
    format_distribution_report,
    load_gold_items,
    merge_gold_files,
    save_gold_items,
    save_gold_set,
    validate_distribution,
)
from food_cpg_intelligence.evaluation.judge import load_judge_scores
from food_cpg_intelligence.evaluation.models import JudgeScore
from food_cpg_intelligence.evaluation.report import compile_report, render_markdown, save_report
from food_cpg_intelligence.logging import configure_logging

app = typer.Typer(name="eval", help="Evaluation framework commands.")


@app.command()
def generate_gold_set(
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Merge and validate gold standard Q&A pairs from category files.

    Gold set generation is done via Claude Code agents (see docs).
    This command merges per-category JSONL files, deduplicates, and validates.
    """
    configure_logging()

    eval_dir = Path(output_dir) if output_dir else settings.resolve_path(settings.evaluation_dir)

    category_files = sorted(eval_dir.glob("gold_*.jsonl"))
    if not category_files:
        typer.echo(f"No gold_*.jsonl files found in {eval_dir}")
        typer.echo("Generate them first using Claude Code agents.")
        raise typer.Exit(code=1)

    typer.echo(f"Merging {len(category_files)} category files...")
    for f in category_files:
        typer.echo(f"  {f.name}")

    items = merge_gold_files(*category_files)
    typer.echo(f"\nMerged: {len(items)} unique items")

    report_text = format_distribution_report(items)
    typer.echo(f"\n{report_text}")

    merged_path = eval_dir / "gold_candidates.jsonl"
    save_gold_items(items, merged_path)

    gold_set = build_gold_set(
        items,
        description=f"Merged from {len(category_files)} category files, {len(items)} items",
    )
    save_gold_set(gold_set, eval_dir / "gold_standard_v1.json")
    typer.echo(f"\nSaved to {eval_dir}")


@app.command()
def validate(
    gold_set_path: str = typer.Option("", help="Path to gold set JSON or JSONL."),
) -> None:
    """Validate distribution and quality of a gold standard set."""
    configure_logging()

    eval_dir = settings.resolve_path(settings.evaluation_dir)
    path = Path(gold_set_path) if gold_set_path else eval_dir / "gold_candidates.jsonl"

    if not path.exists():
        typer.echo(f"File not found: {path}")
        raise typer.Exit(code=1)

    items = load_gold_items(path)
    report_text = format_distribution_report(items)
    typer.echo(report_text)

    validation = validate_distribution(items)
    warnings = validation.get("warnings", [])
    if warnings:
        raise typer.Exit(code=1)


@app.command()
def run_ragas(
    responses_path: str = typer.Option("", help="Path to responses file."),
    gold_set_path: str = typer.Option("", help="Path to gold standard set."),
) -> None:
    """Run RAGAS evaluation metrics."""
    typer.echo("Not yet implemented: run-ragas (Phase 5 — requires RAG pipeline)")
    raise typer.Exit(code=1)


@app.command()
def run_judge(
    responses_path: str = typer.Option("", help="Path to responses file."),
    gold_set_path: str = typer.Option("", help="Path to gold standard set."),
    blind: bool = typer.Option(True, help="Use blind evaluation protocol."),
) -> None:
    """Run LLM-as-judge evaluation via Claude Code agents."""
    typer.echo("Not yet implemented: run-judge (requires system responses)")
    raise typer.Exit(code=1)


@app.command()
def run_all(
    responses_dir: str = typer.Option("", help="Directory with response files."),
    gold_set_path: str = typer.Option("", help="Path to gold standard set."),
) -> None:
    """Run full evaluation pipeline (RAGAS + judge)."""
    typer.echo("Not yet implemented: run-all")
    raise typer.Exit(code=1)


@app.command()
def report(
    results_dir: str = typer.Option("", help="Directory with evaluation results."),
    output: str = typer.Option("", help="Output path for report."),
) -> None:
    """Generate evaluation report from judge scores."""
    configure_logging()

    eval_dir = Path(results_dir) if results_dir else settings.resolve_path(settings.evaluation_dir)

    score_files = sorted(eval_dir.glob("scores_*.jsonl"))
    if not score_files:
        typer.echo(f"No scores_*.jsonl files found in {eval_dir}")
        raise typer.Exit(code=1)

    scores_by_system: dict[str, list[JudgeScore]] = {}
    for sf in score_files:
        system_name = sf.stem.replace("scores_", "")
        scores_by_system[system_name] = load_judge_scores(sf)

    eval_report = compile_report(scores_by_system)
    md = render_markdown(eval_report)
    typer.echo(md)

    out_dir = Path(output) if output else eval_dir
    save_report(eval_report, out_dir)
    typer.echo(f"\nReport saved to {out_dir}")

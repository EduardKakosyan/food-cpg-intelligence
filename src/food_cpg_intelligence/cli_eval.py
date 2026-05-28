"""CLI subcommands for evaluation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer

from food_cpg_intelligence.config import settings
from food_cpg_intelligence.evaluation.gold_set_generator import (
    build_gold_set,
    format_distribution_report,
    load_gold_items,
    load_gold_set,
    merge_gold_files,
    save_gold_items,
    save_gold_set,
    validate_distribution,
)
from food_cpg_intelligence.evaluation.judge import load_judge_scores
from food_cpg_intelligence.evaluation.models import JudgeScore
from food_cpg_intelligence.evaluation.report import compile_report, render_markdown, save_report
from food_cpg_intelligence.evaluation.runner import (
    generate_responses_claude,
    generate_responses_ollama,
    load_eval_responses,
    run_full_pipeline,
    run_judge_scoring,
)
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
def generate_responses(
    system: str = typer.Option(..., help="System: 'finetune' (Ollama) or 'baseline' (Claude)."),
    gold_set_path: str = typer.Option("", help="Path to gold standard JSON."),
    output_path: str = typer.Option("", help="Output JSONL path."),
    model: str = typer.Option("", help="Model name override."),
) -> None:
    """Generate responses from a single system for the gold standard set."""
    configure_logging()

    eval_dir = settings.resolve_path(settings.evaluation_dir)
    gs_path = Path(gold_set_path) if gold_set_path else eval_dir / "gold_standard_v1.json"

    if not gs_path.exists():
        typer.echo(f"Gold set not found: {gs_path}")
        raise typer.Exit(code=1)

    gold_set = load_gold_set(gs_path)
    gold_items = list(gold_set.items)
    typer.echo(f"Loaded {len(gold_items)} gold standard items")

    if system == "finetune":
        model_name = model or settings.ollama_model
        out = Path(output_path) if output_path else eval_dir / "responses_finetune.jsonl"
        typer.echo(f"Generating finetune responses via Ollama ({model_name})...")
        responses = generate_responses_ollama(gold_items, model_name=model_name, output_path=out)
    elif system == "baseline":
        claude_model = model or settings.claude_model
        out = Path(output_path) if output_path else eval_dir / "responses_baseline.jsonl"
        if not settings.anthropic_api_key:
            typer.echo("Error: FCPG_ANTHROPIC_API_KEY not set")
            raise typer.Exit(code=1)
        typer.echo(f"Generating baseline responses via Claude ({claude_model})...")
        responses = generate_responses_claude(
            gold_items,
            model=claude_model,
            output_path=out,
            api_key=settings.anthropic_api_key,
        )
    else:
        typer.echo(f"Unknown system: {system}. Use 'finetune' or 'baseline'.")
        raise typer.Exit(code=1)

    typer.echo(f"Done: {len(responses)} responses saved to {out}")


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
    responses_dir: str = typer.Option("", help="Directory with responses_*.jsonl files."),
    gold_set_path: str = typer.Option("", help="Path to gold standard JSON."),
    output_dir: str = typer.Option("", help="Output directory for scores."),
    judge_model: str = typer.Option("claude-sonnet-4-6", help="Judge model."),
) -> None:
    """Run blind LLM-as-judge evaluation on existing response files."""
    configure_logging()
    from food_cpg_intelligence.evaluation.blind_eval import (
        prepare_blind_batch,
        reveal_results,
    )
    from food_cpg_intelligence.evaluation.judge import save_judge_scores

    eval_dir = settings.resolve_path(settings.evaluation_dir)
    resp_dir = Path(responses_dir) if responses_dir else eval_dir
    gs_path = Path(gold_set_path) if gold_set_path else eval_dir / "gold_standard_v1.json"
    out_dir = Path(output_dir) if output_dir else resp_dir

    if not gs_path.exists():
        typer.echo(f"Gold set not found: {gs_path}")
        raise typer.Exit(code=1)

    if not settings.anthropic_api_key:
        typer.echo("Error: FCPG_ANTHROPIC_API_KEY not set")
        raise typer.Exit(code=1)

    gold_set = load_gold_set(gs_path)
    gold_items = list(gold_set.items)

    # Load all response files
    response_files = sorted(resp_dir.glob("responses_*.jsonl"))
    if not response_files:
        typer.echo(f"No responses_*.jsonl files found in {resp_dir}")
        raise typer.Exit(code=1)

    responses_by_system: dict[str, list] = {}
    for rf in response_files:
        system_name = rf.stem.replace("responses_", "")
        responses_by_system[system_name] = load_eval_responses(rf)
        typer.echo(f"Loaded {len(responses_by_system[system_name])} responses for '{system_name}'")

    # Blind batch
    blind_batch = prepare_blind_batch(gold_items, responses_by_system)
    batch_path = out_dir / "blind_batch.json"
    batch_path.write_text(blind_batch.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"Blind batch: {len(blind_batch.items)} items")

    # Judge
    scores_path = out_dir / "scores_blind.jsonl"
    typer.echo(f"Judging with {judge_model}...")
    blind_scores = run_judge_scoring(
        blind_batch,
        judge_model=judge_model,
        api_key=settings.anthropic_api_key,
        output_path=scores_path,
    )
    typer.echo(f"Scored {len(blind_scores)} items")

    # Unmask and save per-system scores
    scores_by_system = reveal_results(blind_batch, blind_scores)
    for system_name, scores in scores_by_system.items():
        system_path = out_dir / f"scores_{system_name}.jsonl"
        save_judge_scores(scores, system_path)
        typer.echo(f"Saved {len(scores)} scores for '{system_name}'")

    typer.echo("Done. Run 'fcpg eval report' to generate the comparison report.")


@app.command()
def run_all(
    gold_set_path: str = typer.Option("", help="Path to gold standard JSON."),
    output_dir: str = typer.Option("", help="Output directory for all results."),
    ollama_model: str = typer.Option("", help="Ollama model name."),
    claude_model: str = typer.Option("", help="Claude model for baseline."),
    judge_model: str = typer.Option("", help="Claude model for judging."),
) -> None:
    """Run the full evaluation pipeline: generate responses, judge, report."""
    configure_logging()

    eval_dir = settings.resolve_path(settings.evaluation_dir)
    gs_path = Path(gold_set_path) if gold_set_path else eval_dir / "gold_standard_v1.json"

    if not gs_path.exists():
        typer.echo(f"Gold set not found: {gs_path}")
        raise typer.Exit(code=1)

    if not settings.anthropic_api_key:
        typer.echo("Error: FCPG_ANTHROPIC_API_KEY not set")
        raise typer.Exit(code=1)

    # Create timestamped output directory
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) if output_dir else eval_dir / f"benchmark_{timestamp}"

    typer.echo("Running full evaluation pipeline")
    typer.echo(f"  Gold set: {gs_path}")
    typer.echo(f"  Output:   {out}")
    typer.echo(f"  Ollama:   {ollama_model or settings.ollama_model}")
    typer.echo(f"  Claude:   {claude_model or settings.claude_model}")
    typer.echo("")

    result_dir = run_full_pipeline(
        gs_path,
        out,
        ollama_model=ollama_model or settings.ollama_model,
        claude_model=claude_model or settings.claude_model,
        judge_model=judge_model or settings.claude_model,
        api_key=settings.anthropic_api_key,
    )

    typer.echo(f"\nPipeline complete. Results in {result_dir}")
    typer.echo(f"  Report: {result_dir / 'eval_report.md'}")


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

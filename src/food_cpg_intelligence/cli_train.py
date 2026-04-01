"""CLI subcommands for training data generation and fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import structlog
import typer

from food_cpg_intelligence.config import settings
from food_cpg_intelligence.evaluation.gold_set_generator import load_gold_items
from food_cpg_intelligence.logging import configure_logging
from food_cpg_intelligence.training.formatter import save_formatted_dataset
from food_cpg_intelligence.training.models import TrainingTriple, compute_stats
from food_cpg_intelligence.training.quality_filter import run_quality_pipeline

app = typer.Typer(name="train", help="Training data and fine-tuning commands.")


logger = structlog.stdlib.get_logger(__name__)


def _load_triples_jsonl(path: Path) -> list[TrainingTriple]:
    """Load training triples from a JSONL file. Malformed lines are logged and skipped."""
    triples: list[TrainingTriple] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                triples.append(TrainingTriple.model_validate_json(stripped))
            except Exception as exc:
                skipped += 1
                logger.warning("triple_parse_error", path=str(path), line=line_num, error=str(exc))
    if skipped:
        logger.warning("triples_skipped", path=str(path), skipped=skipped)
    return triples


def _save_triples_jsonl(triples: list[TrainingTriple], path: Path) -> None:
    """Save training triples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(t.model_dump_json() + "\n")


@app.command()
def generate_pairs(
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Merge raw batch files from agent-generated training data.

    Training data generation is done via Claude Code agents.
    This command merges raw_batch_*.jsonl files and saves as raw_pairs.jsonl.
    """
    configure_logging()

    train_dir = Path(output_dir) if output_dir else settings.resolve_path(settings.training_dir)
    batch_files = sorted(train_dir.glob("raw_batch_*.jsonl"))

    if not batch_files:
        typer.echo(f"No raw_batch_*.jsonl files found in {train_dir}")
        typer.echo("Generate them first using Claude Code agents.")
        raise typer.Exit(code=1)

    typer.echo(f"Merging {len(batch_files)} batch files...")
    all_triples: list[TrainingTriple] = []
    for bf in batch_files:
        triples = _load_triples_jsonl(bf)
        typer.echo(f"  {bf.name}: {len(triples)} triples")
        all_triples.extend(triples)

    output_path = train_dir / "raw_pairs.jsonl"
    _save_triples_jsonl(all_triples, output_path)
    typer.echo(f"\nMerged: {len(all_triples)} total triples -> {output_path}")


@app.command()
def validate_pairs(
    input_path: str = typer.Option("", help="Path to raw training pairs JSONL."),
    gold_set_path: str = typer.Option(
        "", help="Path to gold standard JSONL for contamination check."
    ),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Run quality filtering on training data."""
    configure_logging()

    train_dir = settings.resolve_path(settings.training_dir)
    eval_dir = settings.resolve_path(settings.evaluation_dir)

    raw_path = Path(input_path) if input_path else train_dir / "raw_pairs.jsonl"
    gold_path = Path(gold_set_path) if gold_set_path else eval_dir / "gold_candidates.jsonl"
    out_dir = Path(output_dir) if output_dir else train_dir

    if not raw_path.exists():
        typer.echo(f"Raw pairs not found: {raw_path}")
        raise typer.Exit(code=1)

    triples = _load_triples_jsonl(raw_path)
    typer.echo(f"Loaded {len(triples)} raw triples")

    gold_items = load_gold_items(gold_path) if gold_path.exists() else []
    if gold_items:
        typer.echo(f"Loaded {len(gold_items)} gold items for contamination check")
    else:
        typer.echo("Warning: No gold set found, skipping contamination check")

    filtered, filter_stats = run_quality_pipeline(triples, gold_items)
    typer.echo(f"\nFilter results: {json.dumps(filter_stats, indent=2)}")

    output_path = out_dir / "filtered_pairs.jsonl"
    _save_triples_jsonl(filtered, output_path)
    typer.echo(f"Saved {len(filtered)} filtered triples -> {output_path}")


@app.command()
def format_data(
    input_path: str = typer.Option("", help="Path to filtered training pairs."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Format training data for Unsloth QLoRA (Qwen 3.5 ChatML)."""
    configure_logging()

    train_dir = settings.resolve_path(settings.training_dir)
    in_path = Path(input_path) if input_path else train_dir / "filtered_pairs.jsonl"
    out_dir = Path(output_dir) if output_dir else train_dir / "formatted"

    if not in_path.exists():
        typer.echo(f"Filtered pairs not found: {in_path}")
        raise typer.Exit(code=1)

    triples = _load_triples_jsonl(in_path)
    typer.echo(f"Loaded {len(triples)} triples")

    train_path, val_path = save_formatted_dataset(triples, out_dir)
    typer.echo(f"Saved: {train_path} and {val_path}")


@app.command()
def stats(
    input_path: str = typer.Option("", help="Path to training dataset JSONL."),
) -> None:
    """Print training dataset statistics."""
    train_dir = settings.resolve_path(settings.training_dir)
    path = Path(input_path) if input_path else train_dir / "filtered_pairs.jsonl"

    if not path.exists():
        typer.echo(f"File not found: {path}")
        raise typer.Exit(code=1)

    triples = _load_triples_jsonl(path)
    ds_stats = compute_stats(triples)
    typer.echo(f"Training Dataset: {ds_stats.total} triples")
    typer.echo(f"  By category:     {json.dumps(ds_stats.by_category, indent=4)}")
    typer.echo(f"  By question type: {json.dumps(ds_stats.by_question_type, indent=4)}")
    typer.echo(f"  Avg instruction:  {ds_stats.avg_instruction_length:.0f} chars")
    typer.echo(f"  Avg response:     {ds_stats.avg_response_length:.0f} chars")


@app.command()
def finetune(
    config: str = typer.Option("configs/training.yaml", help="Training config path."),
    resume_from: str = typer.Option("", help="Checkpoint path to resume from."),
) -> None:
    """Run QLoRA fine-tuning via Unsloth."""
    typer.echo("Not yet implemented: finetune (Phase 4)")
    raise typer.Exit(code=1)


@app.command()
def convert_mlx(
    adapter_path: str = typer.Option(..., help="Path to trained LoRA adapter."),
    output_path: str = typer.Option("", help="Output path for MLX model."),
    quantize: str = typer.Option("4bit", help="Quantization level."),
) -> None:
    """Convert fine-tuned model to MLX format for Apple Silicon."""
    typer.echo("Not yet implemented: convert-mlx (Phase 4)")
    raise typer.Exit(code=1)


@app.command()
def setup_ollama(
    model_path: str = typer.Option(..., help="Path to MLX model."),
    model_name: str = typer.Option("skufood-7b", help="Ollama model name."),
) -> None:
    """Register fine-tuned model with Ollama."""
    typer.echo("Not yet implemented: setup-ollama (Phase 4)")
    raise typer.Exit(code=1)


@app.command()
def eval_checkpoints(
    checkpoints_dir: str = typer.Option(..., help="Directory with training checkpoints."),
    gold_set_path: str = typer.Option(..., help="Path to gold standard set."),
) -> None:
    """Evaluate training checkpoints against gold standard."""
    typer.echo("Not yet implemented: eval-checkpoints (Phase 4)")
    raise typer.Exit(code=1)

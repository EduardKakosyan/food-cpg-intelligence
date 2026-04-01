"""CLI subcommands for training data generation and fine-tuning."""

import typer

app = typer.Typer(name="train", help="Training data and fine-tuning commands.")


@app.command()
def generate_pairs(
    target_count: int = typer.Option(5000, help="Target number of training pairs."),
    input_dir: str = typer.Option("", help="Path to processed documents."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Generate synthetic instruction/response training pairs using Claude."""
    typer.echo("Not yet implemented: generate-pairs")
    raise typer.Exit(code=1)


@app.command()
def validate_pairs(
    input_path: str = typer.Option("", help="Path to raw training pairs."),
    gold_set_path: str = typer.Option(
        "", help="Path to gold standard set for contamination check."
    ),
) -> None:
    """Run quality filtering on training data."""
    typer.echo("Not yet implemented: validate-pairs")
    raise typer.Exit(code=1)


@app.command()
def format(
    input_path: str = typer.Option("", help="Path to filtered training pairs."),
    model_type: str = typer.Option("mistral", help="Model type: 'mistral' or 'llama'."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Format training data for Unsloth (chat template + Arrow)."""
    typer.echo("Not yet implemented: format")
    raise typer.Exit(code=1)


@app.command()
def finetune(
    config: str = typer.Option("configs/training.yaml", help="Training config path."),
    resume_from: str = typer.Option("", help="Checkpoint path to resume from."),
) -> None:
    """Run QLoRA fine-tuning via Unsloth."""
    typer.echo("Not yet implemented: finetune")
    raise typer.Exit(code=1)


@app.command()
def convert_mlx(
    adapter_path: str = typer.Option(..., help="Path to trained LoRA adapter."),
    output_path: str = typer.Option("", help="Output path for MLX model."),
    quantize: str = typer.Option("4bit", help="Quantization level."),
) -> None:
    """Convert fine-tuned model to MLX format for Apple Silicon."""
    typer.echo("Not yet implemented: convert-mlx")
    raise typer.Exit(code=1)


@app.command()
def setup_ollama(
    model_path: str = typer.Option(..., help="Path to MLX model."),
    model_name: str = typer.Option("skufood-7b", help="Ollama model name."),
) -> None:
    """Register fine-tuned model with Ollama."""
    typer.echo("Not yet implemented: setup-ollama")
    raise typer.Exit(code=1)


@app.command()
def eval_checkpoints(
    checkpoints_dir: str = typer.Option(..., help="Directory with training checkpoints."),
    gold_set_path: str = typer.Option(..., help="Path to gold standard set."),
) -> None:
    """Evaluate training checkpoints against gold standard."""
    typer.echo("Not yet implemented: eval-checkpoints")
    raise typer.Exit(code=1)


@app.command()
def stats(
    input_path: str = typer.Option("", help="Path to training dataset."),
) -> None:
    """Print training dataset statistics."""
    typer.echo("Not yet implemented: stats")
    raise typer.Exit(code=1)

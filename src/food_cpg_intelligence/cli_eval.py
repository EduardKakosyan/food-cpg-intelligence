"""CLI subcommands for evaluation."""

import typer

app = typer.Typer(name="eval", help="Evaluation framework commands.")


@app.command()
def generate_gold_set(
    count: int = typer.Option(300, help="Target number of gold standard Q&A pairs."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Generate candidate gold standard Q&A pairs using Claude."""
    typer.echo("Not yet implemented: generate-gold-set")
    raise typer.Exit(code=1)


@app.command()
def run_ragas(
    responses_path: str = typer.Option("", help="Path to responses file."),
    gold_set_path: str = typer.Option("", help="Path to gold standard set."),
) -> None:
    """Run RAGAS evaluation metrics."""
    typer.echo("Not yet implemented: run-ragas")
    raise typer.Exit(code=1)


@app.command()
def run_judge(
    responses_path: str = typer.Option("", help="Path to responses file."),
    gold_set_path: str = typer.Option("", help="Path to gold standard set."),
    blind: bool = typer.Option(True, help="Use blind evaluation protocol."),
) -> None:
    """Run LLM-as-judge evaluation."""
    typer.echo("Not yet implemented: run-judge")
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
    """Generate evaluation report."""
    typer.echo("Not yet implemented: report")
    raise typer.Exit(code=1)

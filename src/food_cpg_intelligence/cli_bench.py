"""CLI subcommands for benchmark orchestration."""

import typer

app = typer.Typer(name="bench", help="Head-to-head benchmark commands.")


@app.command()
def run_blind(
    gold_set_path: str = typer.Option(..., help="Path to gold standard set."),
    rag_store: str = typer.Option("chromadb", help="Vector store for RAG pipeline."),
    finetune_model: str = typer.Option(
        "skufood-7b", help="Ollama model name for fine-tuned model."
    ),
) -> None:
    """Run blind head-to-head benchmark on both systems."""
    typer.echo("Not yet implemented: run-blind")
    raise typer.Exit(code=1)


@app.command()
def compare(
    results_dir: str = typer.Option(..., help="Directory with benchmark results."),
) -> None:
    """Generate side-by-side comparison of systems."""
    typer.echo("Not yet implemented: compare")
    raise typer.Exit(code=1)


@app.command()
def report(
    results_dir: str = typer.Option(..., help="Directory with benchmark results."),
    output: str = typer.Option("benchmark_report.md", help="Output path for report."),
) -> None:
    """Generate final research report."""
    typer.echo("Not yet implemented: report")
    raise typer.Exit(code=1)

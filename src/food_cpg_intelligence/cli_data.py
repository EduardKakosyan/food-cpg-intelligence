"""CLI subcommands for data ingestion and processing."""

import typer

app = typer.Typer(name="data", help="Data ingestion and processing commands.")


@app.command()
def ingest_newsletters(
    input_dir: str = typer.Option("", help="Path to newsletters directory."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Parse all SKUFood newsletter .docx files into structured data."""
    typer.echo("Not yet implemented: ingest-newsletters")
    raise typer.Exit(code=1)


@app.command()
def process_all(
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Run the full ingestion pipeline (newsletters)."""
    typer.echo("Not yet implemented: process-all")
    raise typer.Exit(code=1)


@app.command()
def stats() -> None:
    """Print corpus statistics from the processed data manifest."""
    typer.echo("Not yet implemented: stats")
    raise typer.Exit(code=1)

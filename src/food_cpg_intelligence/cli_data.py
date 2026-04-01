"""CLI subcommands for data ingestion and processing."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from food_cpg_intelligence.config import settings
from food_cpg_intelligence.logging import configure_logging

app = typer.Typer(name="data", help="Data ingestion and processing commands.")


@app.command()
def ingest_newsletters(
    input_dir: str = typer.Option("", help="Path to newsletters directory."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Parse all SKUFood newsletter .docx files into structured data."""
    configure_logging()

    from food_cpg_intelligence.data.cleaner import clean_newsletter
    from food_cpg_intelligence.data.models import ProcessedDocument
    from food_cpg_intelligence.data.newsletter_parser import parse_all_newsletters
    from food_cpg_intelligence.data.serializer import save_processed_documents

    newsletters_path = (
        Path(input_dir) if input_dir else settings.resolve_path(settings.newsletters_dir)
    )
    out_path = Path(output_dir) if output_dir else settings.resolve_path(settings.processed_dir)

    if not newsletters_path.exists():
        typer.echo(f"Error: newsletters directory not found: {newsletters_path}")
        raise typer.Exit(code=1)

    typer.echo(f"Parsing newsletters from: {newsletters_path}")
    newsletters = parse_all_newsletters(newsletters_path, show_progress=True)

    typer.echo(f"Cleaning {len(newsletters)} newsletters...")
    cleaned = [clean_newsletter(n) for n in newsletters]

    typer.echo("Converting to processed documents...")
    docs = [ProcessedDocument.from_newsletter(n) for n in cleaned]

    typer.echo(f"Saving to: {out_path}")
    save_processed_documents(docs, out_path)

    total_words = sum(d.word_count for d in docs)
    typer.echo(f"Done: {len(docs)} documents, {total_words:,} total words")


@app.command()
def process_all(
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Run the full ingestion pipeline (newsletters)."""
    ingest_newsletters(input_dir="", output_dir=output_dir)


@app.command()
def stats() -> None:
    """Print corpus statistics from the processed data manifest."""
    manifest_path = settings.resolve_path(settings.processed_dir) / "corpus_manifest.json"
    if not manifest_path.exists():
        typer.echo(
            f"No manifest found at {manifest_path}. Run 'fcpg data ingest-newsletters' first."
        )
        raise typer.Exit(code=1)

    with manifest_path.open() as f:
        manifest = json.load(f)

    typer.echo(f"Corpus Statistics (generated: {manifest['generated_at']})")
    typer.echo(f"  Total documents: {manifest['total_documents']}")
    typer.echo(f"  Total words:     {manifest['total_words']:,}")
    typer.echo(f"  Avg words/doc:   {manifest['avg_words_per_doc']}")
    typer.echo(f"  By type:         {manifest['by_content_type']}")

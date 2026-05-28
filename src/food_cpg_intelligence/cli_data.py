"""CLI subcommands for data ingestion and processing."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from food_cpg_intelligence.config import settings
from food_cpg_intelligence.logging import configure_logging

app = typer.Typer(name="data", help="Data ingestion and processing commands.")

# Default raw subdirectories under FCPG_RAW_DATA_DIR.
_MEMBERSHIP_SUBDIR = "Membership Content"
_MISC_SUBDIR = "Misc content"


def _build_newsletter_docs(newsletters_path: Path) -> list:
    """Parse + clean newsletters and return ProcessedDocument list."""
    from food_cpg_intelligence.data.cleaner import clean_newsletter
    from food_cpg_intelligence.data.models import ProcessedDocument
    from food_cpg_intelligence.data.newsletter_parser import parse_all_newsletters

    newsletters = parse_all_newsletters(newsletters_path, show_progress=True)
    cleaned = [clean_newsletter(n) for n in newsletters]
    return [ProcessedDocument.from_newsletter(n) for n in cleaned]


def _build_generic_docs(directory: Path, *, content_type: str, prefix: str) -> list:
    """Parse a directory of non-newsletter files into ProcessedDocuments."""
    from food_cpg_intelligence.data.generic_parser import parse_directory

    return parse_directory(
        directory,
        content_type=content_type,  # type: ignore[arg-type]
        prefix=prefix,
        show_progress=True,
    )


@app.command()
def ingest_newsletters(
    input_dir: str = typer.Option("", help="Path to newsletters directory."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Parse all SKUFood newsletter .docx files into structured data."""
    configure_logging()
    from food_cpg_intelligence.data.serializer import save_processed_documents

    newsletters_path = (
        Path(input_dir) if input_dir else settings.resolve_path(settings.newsletters_dir)
    )
    out_path = Path(output_dir) if output_dir else settings.resolve_path(settings.processed_dir)

    if not newsletters_path.exists():
        typer.echo(f"Error: newsletters directory not found: {newsletters_path}")
        raise typer.Exit(code=1)

    typer.echo(f"Parsing newsletters from: {newsletters_path}")
    docs = _build_newsletter_docs(newsletters_path)
    typer.echo(f"Saving to: {out_path}")
    save_processed_documents(docs, out_path)
    total_words = sum(d.word_count for d in docs)
    typer.echo(f"Done: {len(docs)} newsletters, {total_words:,} words")


@app.command()
def ingest_membership(
    input_dir: str = typer.Option("", help="Path to Membership Content directory."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Parse all Membership Content (.docx / .xlsx) into structured data."""
    configure_logging()

    raw_root = settings.resolve_path(settings.raw_data_dir)
    membership_path = Path(input_dir) if input_dir else raw_root / _MEMBERSHIP_SUBDIR
    out_path = Path(output_dir) if output_dir else settings.resolve_path(settings.processed_dir)

    if not membership_path.exists():
        typer.echo(f"Error: membership directory not found: {membership_path}")
        raise typer.Exit(code=1)

    typer.echo(f"Parsing membership content from: {membership_path}")
    docs = _build_generic_docs(membership_path, content_type="membership", prefix="membership")
    out_file = out_path / "corpus_membership.jsonl"
    out_path.mkdir(parents=True, exist_ok=True)
    from food_cpg_intelligence.data.serializer import save_jsonl

    save_jsonl(docs, out_file)
    total_words = sum(d.word_count for d in docs)
    typer.echo(f"Done: {len(docs)} membership docs, {total_words:,} words -> {out_file}")


@app.command()
def ingest_misc(
    input_dir: str = typer.Option("", help="Path to Misc content directory."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Parse Misc Content (mixed .docx/.pdf/.pptx/.xlsx) into structured data."""
    configure_logging()
    from food_cpg_intelligence.data.serializer import save_jsonl

    raw_root = settings.resolve_path(settings.raw_data_dir)
    misc_path = Path(input_dir) if input_dir else raw_root / _MISC_SUBDIR
    out_path = Path(output_dir) if output_dir else settings.resolve_path(settings.processed_dir)

    if not misc_path.exists():
        typer.echo(f"Error: misc directory not found: {misc_path}")
        raise typer.Exit(code=1)

    typer.echo(f"Parsing misc content from: {misc_path}")
    docs = _build_generic_docs(misc_path, content_type="reference", prefix="reference")
    out_file = out_path / "corpus_misc.jsonl"
    out_path.mkdir(parents=True, exist_ok=True)
    save_jsonl(docs, out_file)
    total_words = sum(d.word_count for d in docs)
    typer.echo(f"Done: {len(docs)} misc docs, {total_words:,} words -> {out_file}")


@app.command()
def ingest_all(
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Ingest newsletters + membership + misc, then write merged corpus + manifest."""
    configure_logging()
    from food_cpg_intelligence.data.serializer import save_processed_documents

    out_path = Path(output_dir) if output_dir else settings.resolve_path(settings.processed_dir)
    raw_root = settings.resolve_path(settings.raw_data_dir)
    newsletters_path = settings.resolve_path(settings.newsletters_dir)
    membership_path = raw_root / _MEMBERSHIP_SUBDIR
    misc_path = raw_root / _MISC_SUBDIR

    all_docs: list = []
    by_type: dict[str, int] = {}

    if newsletters_path.exists():
        typer.echo(f"[1/3] newsletters: {newsletters_path}")
        docs = _build_newsletter_docs(newsletters_path)
        all_docs.extend(docs)
        by_type["newsletter"] = len(docs)
    else:
        typer.echo(f"[1/3] skipped (not found): {newsletters_path}")

    if membership_path.exists():
        typer.echo(f"[2/3] membership: {membership_path}")
        docs = _build_generic_docs(membership_path, content_type="membership", prefix="membership")
        all_docs.extend(docs)
        by_type["membership"] = len(docs)
    else:
        typer.echo(f"[2/3] skipped (not found): {membership_path}")

    if misc_path.exists():
        typer.echo(f"[3/3] misc: {misc_path}")
        docs = _build_generic_docs(misc_path, content_type="reference", prefix="reference")
        all_docs.extend(docs)
        by_type["reference"] = len(docs)
    else:
        typer.echo(f"[3/3] skipped (not found): {misc_path}")

    if not all_docs:
        typer.echo("No documents parsed. Nothing to write.")
        raise typer.Exit(code=1)

    typer.echo(f"Writing merged corpus to: {out_path}")
    save_processed_documents(all_docs, out_path)
    total_words = sum(d.word_count for d in all_docs)
    typer.echo(f"Done: {len(all_docs)} documents ({by_type}), {total_words:,} total words")


@app.command()
def process_all(
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Alias for `ingest-all` — kept for backwards compatibility."""
    ingest_all(output_dir=output_dir)


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

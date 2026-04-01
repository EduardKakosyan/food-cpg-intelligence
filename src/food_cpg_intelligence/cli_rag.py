"""CLI subcommands for RAG pipeline."""

import typer

app = typer.Typer(name="rag", help="RAG pipeline commands.")


@app.command()
def index(
    input_dir: str = typer.Option("data/processed", help="Path to processed documents."),
    store: str = typer.Option("chromadb", help="Vector store backend: 'chromadb' or 'qdrant'."),
) -> None:
    """Embed and index the processed corpus into a vector store."""
    typer.echo("Not yet implemented: index")
    raise typer.Exit(code=1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask."),
    top_k: int = typer.Option(5, help="Number of context chunks to retrieve."),
) -> None:
    """Run a single RAG query interactively."""
    typer.echo("Not yet implemented: query")
    raise typer.Exit(code=1)


@app.command()
def evaluate(
    store: str = typer.Option("", help="Vector store to evaluate. Empty = compare both."),
) -> None:
    """Evaluate vector store(s) — compare ChromaDB vs Qdrant."""
    typer.echo("Not yet implemented: evaluate")
    raise typer.Exit(code=1)

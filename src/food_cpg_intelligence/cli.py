"""CLI entrypoint for the Food CPG Intelligence platform."""

import typer

from food_cpg_intelligence.cli_bench import app as bench_app
from food_cpg_intelligence.cli_data import app as data_app
from food_cpg_intelligence.cli_eval import app as eval_app
from food_cpg_intelligence.cli_rag import app as rag_app
from food_cpg_intelligence.cli_train import app as train_app

app = typer.Typer(
    name="fcpg",
    help="Food & CPG Product Intelligence Platform CLI",
    add_completion=False,
)

# Register subcommand groups
app.add_typer(data_app)
app.add_typer(eval_app)
app.add_typer(train_app)
app.add_typer(rag_app)
app.add_typer(bench_app)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """Food & CPG Product Intelligence Platform."""
    if version:
        from food_cpg_intelligence import __version__

        typer.echo(f"food-cpg-intelligence v{__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


if __name__ == "__main__":
    app()

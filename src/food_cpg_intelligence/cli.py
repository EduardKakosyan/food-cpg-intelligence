"""CLI entrypoint for the Food CPG Intelligence platform."""

import typer

app = typer.Typer(
    name="fcpg",
    help="Food & CPG Product Intelligence Platform CLI",
    add_completion=False,
)


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

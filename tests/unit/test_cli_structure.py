"""Tests for CLI subcommand registration."""

from typer.testing import CliRunner

from food_cpg_intelligence.cli import app

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "food-cpg-intelligence" in result.output


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "data" in result.output
    assert "eval" in result.output
    assert "train" in result.output
    assert "rag" in result.output
    assert "bench" in result.output


def test_data_help() -> None:
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "ingest-newsletters" in result.output
    assert "process-all" in result.output
    assert "stats" in result.output


def test_eval_help() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "generate-gold-set" in result.output
    assert "run-ragas" in result.output
    assert "run-judge" in result.output
    assert "run-all" in result.output
    assert "report" in result.output


def test_train_help() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "generate-pairs" in result.output
    assert "validate-pairs" in result.output
    assert "finetune" in result.output
    assert "convert-mlx" in result.output
    assert "setup-ollama" in result.output


def test_rag_help() -> None:
    result = runner.invoke(app, ["rag", "--help"])
    assert result.exit_code == 0
    assert "index" in result.output
    assert "query" in result.output
    assert "evaluate" in result.output


def test_bench_help() -> None:
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "run-blind" in result.output
    assert "compare" in result.output
    assert "report" in result.output

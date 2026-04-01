"""Tests for MLX data preparation."""

from pathlib import Path

from food_cpg_intelligence.training.formatter import SYSTEM_PROMPT
from food_cpg_intelligence.training.mlx_data_prep import (
    prepare_mlx_data,
    triple_to_chat_messages,
)
from food_cpg_intelligence.training.models import TrainingTriple


def _make_triple(
    instruction: str = "How to price?",
    context: str = "Peter discusses pricing.",
    response: str = "Focus on margin and value.",
) -> TrainingTriple:
    return TrainingTriple(
        triple_id="t1",
        instruction=instruction,
        context=context,
        response=response,
    )


def test_triple_to_chat_messages_structure() -> None:
    t = _make_triple()
    result = triple_to_chat_messages(t)
    assert "messages" in result
    msgs = result["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert msgs[2]["role"] == "assistant"


def test_triple_to_chat_messages_content() -> None:
    t = _make_triple(instruction="Q?", context="CTX", response="A!")
    msgs = triple_to_chat_messages(t)["messages"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "Q?" in msgs[1]["content"]
    assert "CTX" in msgs[1]["content"]
    assert msgs[2]["content"] == "A!"


def test_triple_to_chat_messages_no_context() -> None:
    t = _make_triple(context="")
    msgs = triple_to_chat_messages(t)["messages"]
    assert "Context:" not in msgs[1]["content"]


def test_prepare_mlx_data(tmp_path: Path) -> None:
    # Create a filtered_pairs.jsonl
    triples = [
        TrainingTriple(
            triple_id=f"t{i}",
            instruction=f"Q{i}?",
            context=f"C{i}",
            response=f"A{i} with enough content to be useful for training purposes.",
        )
        for i in range(20)
    ]
    input_path = tmp_path / "filtered_pairs.jsonl"
    with input_path.open("w") as f:
        for t in triples:
            f.write(t.model_dump_json() + "\n")

    out_dir = tmp_path / "mlx"
    train_path, val_path = prepare_mlx_data(input_path, out_dir)

    assert train_path.exists()
    assert val_path.exists()
    assert train_path.name == "train.jsonl"
    assert val_path.name == "valid.jsonl"

    # Check content is chat messages format
    import json

    with train_path.open() as f:
        first = json.loads(f.readline())
    assert "messages" in first
    assert len(first["messages"]) == 3
    assert first["messages"][0]["role"] == "system"

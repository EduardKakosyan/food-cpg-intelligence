"""Tests for gold set loader/validator."""

from pathlib import Path

from food_cpg_intelligence.evaluation.gold_set_generator import (
    build_gold_set,
    deduplicate,
    format_distribution_report,
    generate_question_id,
    load_gold_items,
    load_gold_set,
    merge_gold_files,
    save_gold_items,
    save_gold_set,
    validate_distribution,
)
from food_cpg_intelligence.evaluation.models import Category, GoldStandardItem


def _make_items(n: int = 5, category: Category = Category.GENERAL_CPG) -> list[GoldStandardItem]:
    return [
        GoldStandardItem(
            question_id=f"{category.value}-{i:03d}",
            question=f"Question {i} about {category.value}?",
            reference_answer=f"Answer {i}.",
            category=category,
            difficulty="easy" if i % 3 == 0 else "medium" if i % 3 == 1 else "hard",
            source_doc_ids=(f"newsletter-{i}",),
        )
        for i in range(n)
    ]


def test_save_and_load_gold_items(tmp_path: Path) -> None:
    items = _make_items(3)
    path = tmp_path / "test.jsonl"
    save_gold_items(items, path)
    loaded = load_gold_items(path)
    assert len(loaded) == 3
    assert loaded[0].question_id == items[0].question_id


def test_build_gold_set() -> None:
    items = _make_items(3)
    gs = build_gold_set(items, version="2.0", description="test set")
    assert len(gs.items) == 3
    assert gs.version == "2.0"
    assert gs.created_at is not None


def test_save_and_load_gold_set(tmp_path: Path) -> None:
    items = _make_items(3)
    gs = build_gold_set(items)
    path = tmp_path / "gold_set.json"
    save_gold_set(gs, path)
    loaded = load_gold_set(path)
    assert len(loaded.items) == 3
    assert loaded.version == gs.version


def test_generate_question_id() -> None:
    qid = generate_question_id(Category.TRADE_SHOWS)
    assert qid.startswith("trade_shows-")
    assert len(qid) > len("trade_shows-")


def test_validate_distribution_good() -> None:
    items: list[GoldStandardItem] = []
    for cat in Category:
        count = 25 if cat == Category.UNANSWERABLE else 50
        items.extend(_make_items(count, cat))
    report = validate_distribution(items)
    assert report["total"] == 275
    assert len(report["warnings"]) == 0  # type: ignore[arg-type]


def test_validate_distribution_warnings() -> None:
    items = _make_items(5, Category.GENERAL_CPG)
    report = validate_distribution(items)
    warnings: list[str] = report["warnings"]  # type: ignore[assignment]
    assert len(warnings) > 0
    assert any("Missing category" in w for w in warnings)


def test_deduplicate_removes_exact_dupes() -> None:
    items = [
        GoldStandardItem(
            question_id="a",
            question="What is X?",
            reference_answer="A",
            category=Category.GENERAL_CPG,
        ),
        GoldStandardItem(
            question_id="b",
            question="what is x?",  # same question, different case
            reference_answer="B",
            category=Category.GENERAL_CPG,
        ),
        GoldStandardItem(
            question_id="c",
            question="What is Y?",
            reference_answer="C",
            category=Category.GENERAL_CPG,
        ),
    ]
    result = deduplicate(items)
    assert len(result) == 2
    assert result[0].question_id == "a"
    assert result[1].question_id == "c"


def test_merge_gold_files(tmp_path: Path) -> None:
    items_a = _make_items(3, Category.RETAILER_STRATEGY)
    items_b = _make_items(3, Category.PRICING_PROMOTION)
    path_a = tmp_path / "gold_retailer.jsonl"
    path_b = tmp_path / "gold_pricing.jsonl"
    save_gold_items(items_a, path_a)
    save_gold_items(items_b, path_b)

    merged = merge_gold_files(path_a, path_b)
    assert len(merged) == 6


def test_format_distribution_report() -> None:
    items = _make_items(10, Category.GENERAL_CPG)
    report = format_distribution_report(items)
    assert "general_cpg" in report
    assert "10" in report

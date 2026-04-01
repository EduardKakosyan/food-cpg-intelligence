"""Tests for training data quality filter."""

from food_cpg_intelligence.evaluation.models import Category, GoldStandardItem
from food_cpg_intelligence.training.models import TrainingTriple
from food_cpg_intelligence.training.quality_filter import (
    balance_categories,
    check_contamination,
    deduplicate,
    filter_by_length,
    run_quality_pipeline,
)


def _make_triple(
    triple_id: str = "t1",
    instruction: str = "How should I approach retailers?",
    response: str = "Here is practical advice about approaching Canadian retailers with your product.",
    category: str = "general_cpg",
) -> TrainingTriple:
    return TrainingTriple(
        triple_id=triple_id,
        instruction=instruction,
        context="Some context.",
        response=response,
        category=category,
    )


def test_filter_by_length_keeps_good() -> None:
    t = _make_triple(response="A" * 100)
    result = filter_by_length([t])
    assert len(result) == 1


def test_filter_by_length_removes_short_response() -> None:
    t = _make_triple(response="Too short")
    result = filter_by_length([t])
    assert len(result) == 0


def test_filter_by_length_removes_long_response() -> None:
    t = _make_triple(response="A" * 2001)
    result = filter_by_length([t])
    assert len(result) == 0


def test_filter_by_length_removes_short_instruction() -> None:
    t = _make_triple(instruction="Hi?")
    result = filter_by_length([t])
    assert len(result) == 0


def test_deduplicate_removes_exact() -> None:
    t1 = _make_triple(triple_id="t1", instruction="How to price?")
    t2 = _make_triple(triple_id="t2", instruction="how to price?")  # same normalized
    t3 = _make_triple(triple_id="t3", instruction="Different question?")
    result, removed = deduplicate([t1, t2, t3])
    assert len(result) == 2
    assert removed == 1


def test_deduplicate_no_dupes() -> None:
    t1 = _make_triple(triple_id="t1", instruction="Question A?")
    t2 = _make_triple(triple_id="t2", instruction="Question B?")
    result, removed = deduplicate([t1, t2])
    assert len(result) == 2
    assert removed == 0


def test_check_contamination_removes_matches() -> None:
    t1 = _make_triple(instruction="How should I prepare for a Loblaw category review?")
    t2 = _make_triple(instruction="Unrelated question about pricing?")
    gold = [
        GoldStandardItem(
            question_id="g1",
            question="How should I prepare for a Loblaw category review?",
            reference_answer="Focus on data.",
            category=Category.RETAILER_STRATEGY,
        ),
    ]
    clean, removed = check_contamination([t1, t2], gold)
    assert len(clean) == 1
    assert clean[0].instruction == "Unrelated question about pricing?"
    assert removed == 1


def test_check_contamination_case_insensitive() -> None:
    t = _make_triple(instruction="how should i prepare for a loblaw category review?")
    gold = [
        GoldStandardItem(
            question_id="g1",
            question="How should I prepare for a Loblaw category review?",
            reference_answer="A",
            category=Category.RETAILER_STRATEGY,
        ),
    ]
    clean, removed = check_contamination([t], gold)
    assert len(clean) == 0
    assert removed == 1


def test_check_contamination_empty_gold() -> None:
    t = _make_triple()
    clean, removed = check_contamination([t], [])
    assert len(clean) == 1
    assert removed == 0


def test_balance_categories_caps_overrepresented() -> None:
    triples = [_make_triple(triple_id=f"t{i}", category="general_cpg") for i in range(80)]
    triples += [
        _make_triple(triple_id=f"t{80 + i}", category="retailer_strategy") for i in range(20)
    ]
    # 100 total, 30% cap = 30 per category max
    result = balance_categories(triples, max_fraction=0.30)
    cat_counts = {}
    for t in result:
        cat_counts[t.category] = cat_counts.get(t.category, 0) + 1
    assert cat_counts["general_cpg"] <= 30
    assert cat_counts["retailer_strategy"] == 20  # already under cap


def test_balance_categories_no_change_when_balanced() -> None:
    triples = [_make_triple(triple_id=f"t{i}", category=f"cat{i % 5}") for i in range(100)]
    result = balance_categories(triples, max_fraction=0.30)
    assert len(result) == 100  # all categories at 20% < 30%


def test_balance_categories_empty_input() -> None:
    assert balance_categories([]) == []


def test_balance_categories_empty_category_defaults_to_unknown() -> None:
    triples = [_make_triple(triple_id=f"t{i}", category="") for i in range(5)]
    # 5 items, all in "unknown", max_fraction=0.50 -> max_per_cat = max(1, int(5*0.5)) = 2
    result = balance_categories(triples, max_fraction=0.50)
    assert len(result) == 2


def test_run_quality_pipeline_end_to_end() -> None:
    triples = [
        _make_triple(triple_id=f"t{i}", response="A" * 100, category="general_cpg")
        for i in range(10)
    ]
    # Add a short one that should be filtered
    triples.append(_make_triple(triple_id="short", response="X"))
    # Add a dupe
    triples.append(
        _make_triple(triple_id="dupe", instruction=triples[0].instruction, response="A" * 100)
    )

    gold = [
        GoldStandardItem(
            question_id="g1",
            question="Contaminated question?",
            reference_answer="A",
            category=Category.GENERAL_CPG,
        ),
    ]

    _filtered, stats = run_quality_pipeline(triples, gold)
    assert stats["input"] == 12
    assert stats["output"] <= 10  # short + dupe removed at minimum
    assert stats["dedup_removed"] >= 1

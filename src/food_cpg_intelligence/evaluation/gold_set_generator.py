"""Load, save, validate, and manage the gold standard test set.

Generation is done externally (via Claude Code agents or manual curation).
This module handles the lifecycle of gold set files: loading from JSONL,
validating schema/distribution, deduplication, and saving.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

import structlog

from food_cpg_intelligence.evaluation.models import (
    Category,
    GoldStandardItem,
    GoldStandardSet,
)

logger = structlog.stdlib.get_logger(__name__)


def load_gold_items(path: Path) -> list[GoldStandardItem]:
    """Load gold standard items from a JSONL file.

    Malformed lines are logged and skipped rather than aborting the entire load.
    """
    items: list[GoldStandardItem] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                items.append(GoldStandardItem.model_validate_json(stripped))
            except Exception as exc:
                skipped += 1
                logger.warning(
                    "gold_item_parse_error", path=str(path), line=line_num, error=str(exc)
                )
    if skipped:
        logger.warning("gold_items_skipped", path=str(path), skipped=skipped)
    logger.info("loaded_gold_items", path=str(path), count=len(items))
    return items


def save_gold_items(items: list[GoldStandardItem], path: Path) -> Path:
    """Save gold standard items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    logger.info("saved_gold_items", path=str(path), count=len(items))
    return path


def build_gold_set(
    items: list[GoldStandardItem],
    *,
    version: str = "1.0",
    description: str = "",
) -> GoldStandardSet:
    """Build a GoldStandardSet from a list of items."""
    return GoldStandardSet(
        items=tuple(items),
        version=version,
        created_at=datetime.now(tz=UTC),
        description=description,
    )


def save_gold_set(gold_set: GoldStandardSet, path: Path) -> Path:
    """Save a GoldStandardSet as a JSON file (single object, not JSONL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(gold_set.model_dump_json(indent=2))
    logger.info("saved_gold_set", path=str(path), count=len(gold_set.items))
    return path


def load_gold_set(path: Path) -> GoldStandardSet:
    """Load a GoldStandardSet from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return GoldStandardSet.model_validate_json(f.read())


def generate_question_id(category: Category) -> str:
    """Generate a unique question ID with category prefix."""
    short_id = uuid.uuid4().hex[:8]
    return f"{category.value}-{short_id}"


def validate_distribution(
    items: list[GoldStandardItem],
) -> dict[str, object]:
    """Validate the category and difficulty distribution of gold set items.

    Returns a report dict with distribution stats and any warnings.
    """
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    verified_count = 0

    for item in items:
        cat = item.category.value
        category_counts[cat] = category_counts.get(cat, 0) + 1
        difficulty_counts[item.difficulty] = difficulty_counts.get(item.difficulty, 0) + 1
        if item.verified:
            verified_count += 1

    warnings: list[str] = []

    # Check all categories are represented
    for cat in Category:
        if cat.value not in category_counts:
            warnings.append(f"Missing category: {cat.value}")

    # Check minimum counts
    for cat, count in category_counts.items():
        if cat != Category.UNANSWERABLE.value and count < 20:
            warnings.append(f"Low count for {cat}: {count} (target: 50)")
    unanswerable = category_counts.get(Category.UNANSWERABLE.value, 0)
    if unanswerable < 10:
        warnings.append(f"Low unanswerable count: {unanswerable} (target: 25)")

    report: dict[str, object] = {
        "total": len(items),
        "verified": verified_count,
        "unverified": len(items) - verified_count,
        "by_category": category_counts,
        "by_difficulty": difficulty_counts,
        "warnings": warnings,
    }

    for warning in warnings:
        logger.warning("gold_set_distribution", warning=warning)

    return report


def deduplicate(items: list[GoldStandardItem]) -> list[GoldStandardItem]:
    """Remove duplicate items based on exact question text match."""
    seen: set[str] = set()
    unique: list[GoldStandardItem] = []
    for item in items:
        normalized = item.question.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(item)
    removed = len(items) - len(unique)
    if removed:
        logger.info("dedup_removed", removed=removed, remaining=len(unique))
    return unique


def merge_gold_files(*paths: Path) -> list[GoldStandardItem]:
    """Load and merge multiple JSONL gold set files, deduplicating."""
    all_items: list[GoldStandardItem] = []
    for path in paths:
        if path.exists():
            all_items.extend(load_gold_items(path))
    return deduplicate(all_items)


def format_distribution_report(items: list[GoldStandardItem]) -> str:
    """Format a human-readable distribution report string."""
    report = validate_distribution(items)
    lines = [
        f"Gold Standard Set: {report['total']} items "
        f"({report['verified']} verified, {report['unverified']} unverified)",
        "",
        "By Category:",
    ]
    by_cat: dict[str, int] = report["by_category"]  # type: ignore[assignment]
    for cat in Category:
        count = by_cat.get(cat.value, 0)
        lines.append(f"  {cat.value:25s} {count:4d}")

    lines.append("")
    lines.append("By Difficulty:")
    by_diff: dict[str, int] = report["by_difficulty"]  # type: ignore[assignment]
    for diff in ("easy", "medium", "hard"):
        count = by_diff.get(diff, 0)
        lines.append(f"  {diff:25s} {count:4d}")

    warnings: list[str] = report["warnings"]  # type: ignore[assignment]
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)

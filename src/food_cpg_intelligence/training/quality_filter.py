"""Quality filtering for training data: length, dedup, contamination, balance."""

from __future__ import annotations

import structlog

from food_cpg_intelligence.evaluation.models import GoldStandardItem
from food_cpg_intelligence.training.models import TrainingTriple

logger = structlog.stdlib.get_logger(__name__)

MIN_RESPONSE_CHARS = 50
MAX_RESPONSE_CHARS = 2000
MIN_INSTRUCTION_CHARS = 10
MAX_CATEGORY_FRACTION = 0.30


def filter_by_length(triples: list[TrainingTriple]) -> list[TrainingTriple]:
    """Remove triples with responses outside acceptable length range."""
    filtered = [
        t
        for t in triples
        if MIN_RESPONSE_CHARS <= len(t.response) <= MAX_RESPONSE_CHARS
        and len(t.instruction) >= MIN_INSTRUCTION_CHARS
    ]
    removed = len(triples) - len(filtered)
    if removed:
        logger.info("length_filter", removed=removed, remaining=len(filtered))
    return filtered


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for deduplication comparison."""
    return " ".join(text.strip().lower().split())


def deduplicate(triples: list[TrainingTriple]) -> tuple[list[TrainingTriple], int]:
    """Remove duplicate triples based on normalized instruction text.

    Returns the deduplicated list and the count of removed duplicates.
    """
    seen: set[str] = set()
    unique: list[TrainingTriple] = []
    for t in triples:
        key = _normalize_for_dedup(t.instruction)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    removed = len(triples) - len(unique)
    if removed:
        logger.info("dedup_removed", removed=removed, remaining=len(unique))
    return unique, removed


def check_contamination(
    triples: list[TrainingTriple],
    gold_items: list[GoldStandardItem],
) -> tuple[list[TrainingTriple], int]:
    """Remove training triples whose questions match gold standard questions.

    Uses normalized exact-match comparison. Any training instruction that
    matches a gold set question (after normalization) is removed to prevent
    data leakage into the evaluation set.

    Returns the clean list and count of contaminated items removed.
    """
    gold_questions = {_normalize_for_dedup(item.question) for item in gold_items}

    clean: list[TrainingTriple] = []
    for t in triples:
        normalized = _normalize_for_dedup(t.instruction)
        if normalized not in gold_questions:
            clean.append(t)

    removed = len(triples) - len(clean)
    if removed:
        logger.warning("contamination_removed", removed=removed, remaining=len(clean))
    return clean, removed


def balance_categories(
    triples: list[TrainingTriple],
    *,
    max_fraction: float = MAX_CATEGORY_FRACTION,
) -> list[TrainingTriple]:
    """Cap any single category at max_fraction of total items.

    Excess items from over-represented categories are trimmed (last in, first out).
    """
    total = len(triples)
    if total == 0:
        return triples

    max_per_cat = max(1, int(total * max_fraction))

    by_cat: dict[str, list[TrainingTriple]] = {}
    for t in triples:
        cat = t.category or "unknown"
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(t)

    balanced: list[TrainingTriple] = []
    trimmed = 0
    for _cat, items in by_cat.items():
        if len(items) > max_per_cat:
            trimmed += len(items) - max_per_cat
            balanced.extend(items[:max_per_cat])
        else:
            balanced.extend(items)

    if trimmed:
        logger.info("category_balance_trimmed", trimmed=trimmed, remaining=len(balanced))
    return balanced


def run_quality_pipeline(
    triples: list[TrainingTriple],
    gold_items: list[GoldStandardItem],
) -> tuple[list[TrainingTriple], dict[str, int]]:
    """Run the full quality filtering pipeline.

    Steps: length filter -> dedup -> contamination check -> category balance.

    Returns the filtered list and a stats dict with removal counts.
    """
    stats: dict[str, int] = {"input": len(triples)}

    filtered = filter_by_length(triples)
    stats["after_length"] = len(filtered)

    filtered, dedup_count = deduplicate(filtered)
    stats["dedup_removed"] = dedup_count

    filtered, contam_count = check_contamination(filtered, gold_items)
    stats["contamination_removed"] = contam_count

    filtered = balance_categories(filtered)
    stats["after_balance"] = len(filtered)
    stats["output"] = len(filtered)

    logger.info("quality_pipeline_complete", **stats)
    return filtered, stats

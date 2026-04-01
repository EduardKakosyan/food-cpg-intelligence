"""Data models for the training data pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainingTriple(BaseModel, frozen=True):
    """A single instruction/context/response triple for fine-tuning."""

    triple_id: str
    instruction: str
    context: str
    response: str
    source_doc_id: str = ""
    category: str = ""
    question_type: Literal["factual", "analytical", "advice", "scenario"] = "factual"
    quality_score: float | None = None


class DatasetStats(BaseModel, frozen=True):
    """Statistics for a training dataset."""

    total: int = 0
    by_category: dict[str, int] = Field(default_factory=dict)
    by_question_type: dict[str, int] = Field(default_factory=dict)
    avg_instruction_length: float = 0.0
    avg_response_length: float = 0.0
    dedup_removed: int = 0
    contamination_removed: int = 0


def compute_stats(
    triples: list[TrainingTriple],
    *,
    dedup_removed: int = 0,
    contamination_removed: int = 0,
) -> DatasetStats:
    """Compute statistics for a list of training triples."""
    if not triples:
        return DatasetStats()

    by_cat: dict[str, int] = {}
    by_type: dict[str, int] = {}
    total_inst_len = 0
    total_resp_len = 0

    for t in triples:
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
        by_type[t.question_type] = by_type.get(t.question_type, 0) + 1
        total_inst_len += len(t.instruction)
        total_resp_len += len(t.response)

    return DatasetStats(
        total=len(triples),
        by_category=by_cat,
        by_question_type=by_type,
        avg_instruction_length=round(total_inst_len / len(triples), 1),
        avg_response_length=round(total_resp_len / len(triples), 1),
        dedup_removed=dedup_removed,
        contamination_removed=contamination_removed,
    )

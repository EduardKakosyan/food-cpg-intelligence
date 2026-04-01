"""Data models for the three-layer evaluation framework."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Category(StrEnum):
    """Question categories for the gold standard test set."""

    RETAILER_STRATEGY = "retailer_strategy"
    PRICING_PROMOTION = "pricing_promotion"
    TRADE_SHOWS = "trade_shows"
    PRODUCT_LAUNCH = "product_launch"
    GENERAL_CPG = "general_cpg"
    UNANSWERABLE = "unanswerable"


class GoldStandardItem(BaseModel, frozen=True):
    """A single verified Q&A pair in the gold standard test set."""

    question_id: str
    question: str
    reference_answer: str
    category: Category
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source_doc_ids: tuple[str, ...] = ()
    requires_multi_doc: bool = False
    verified: bool = False


class GoldStandardSet(BaseModel, frozen=True):
    """The complete gold standard test set with metadata."""

    items: tuple[GoldStandardItem, ...] = ()
    version: str = "1.0"
    created_at: datetime | None = None
    description: str = ""

    @property
    def category_counts(self) -> dict[str, int]:
        """Count items per category."""
        counts: dict[str, int] = {}
        for item in self.items:
            key = item.category.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def difficulty_counts(self) -> dict[str, int]:
        """Count items per difficulty level."""
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.difficulty] = counts.get(item.difficulty, 0) + 1
        return counts


class EvalResponse(BaseModel, frozen=True):
    """A system's response to a gold standard question."""

    question_id: str
    system: str  # "finetune", "rag", "baseline"
    response: str
    latency_ms: float = 0.0
    context_used: tuple[str, ...] | None = None


class JudgeScore(BaseModel, frozen=True):
    """LLM-as-judge scoring for a single response."""

    question_id: str
    system: str
    accuracy: float = 0.0
    grounding: float = 0.0
    completeness: float = 0.0
    voice_fidelity: float = 0.0
    hallucination_resistance: float = 0.0
    overall: float = 0.0
    reasoning: str = ""

    @property
    def dimension_scores(self) -> dict[str, float]:
        """Return all 5 dimension scores as a dict."""
        return {
            "accuracy": self.accuracy,
            "grounding": self.grounding,
            "completeness": self.completeness,
            "voice_fidelity": self.voice_fidelity,
            "hallucination_resistance": self.hallucination_resistance,
        }


class BlindItem(BaseModel, frozen=True):
    """A single blinded item for judge evaluation."""

    blind_id: str
    question_id: str
    question: str
    reference_answer: str
    response: str
    # system identity is hidden — stored in the mapping only
    category: Category


class BlindBatch(BaseModel, frozen=True):
    """A batch of blinded items ready for judging, with a hidden mapping."""

    items: tuple[BlindItem, ...] = ()
    # Maps blind_id -> system name (hidden from judge)
    id_to_system: dict[str, str] = Field(default_factory=dict)
    shuffle_seed: int = 42


class EvalReport(BaseModel, frozen=True):
    """Aggregated evaluation results for reporting."""

    systems: tuple[str, ...] = ()
    scores_by_system: dict[str, dict[str, float]] = Field(default_factory=dict)
    scores_by_category: dict[str, dict[str, dict[str, float]]] = Field(default_factory=dict)
    head_to_head: dict[str, dict[str, int]] = Field(default_factory=dict)
    confidence_intervals: dict[str, dict[str, tuple[float, float]]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

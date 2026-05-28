"""Quality filters for diarized Peter segments.

Pure functions and a small dataclass of thresholds — no heavy ML deps. Each filter
returns a bool so we can record per-filter pass/fail counters in a `FilterReport`
and surface why segments were dropped.

Thresholds default to the values recommended in `docs/research-voice-rag-pivot.md`
section 5 (Audio -> SFT data pipeline).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from food_cpg_intelligence.ingest.audio.schemas import FilterReport, PeterSegment

# Conservative filler set. We do NOT include "like" or "actually" — both have
# legitimate uses in Peter's speech ("I'd like to share...", "actually a great
# question"). Filler density is a coarse signal; we lean on the topic gate to
# remove banter segments more decisively.
_FILLER_WORDS: frozenset[str] = frozenset({"um", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm"})
_FILLER_PHRASES: tuple[str, ...] = ("you know", "kind of", "sort of")

_WORD_RE = re.compile(r"\b[\w']+\b")
_REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1){3,}\b", re.IGNORECASE)
"""A word repeated 4+ times in a row — a Whisper hallucination signature."""


@dataclass(frozen=True)
class FilterThresholds:
    """Tunable filter thresholds. Defaults come from the research synthesis."""

    min_duration_sec: float = 3.0
    max_duration_sec: float = 90.0
    min_asr_confidence: float = 0.6
    max_no_speech_prob: float = 0.3
    max_compression_ratio: float = 2.4
    max_filler_ratio: float = 0.15
    max_overlap_ratio: float = 0.10
    min_speaker_confidence: float = 0.55
    min_topic_similarity: float = 0.0
    """Absolute cosine threshold for topic relevance. Set 0 to disable absolute gating.

    Relative gating (keep top X% of the corpus) is applied separately by the orchestrator
    once all segment similarities are known.
    """


def duration_ok(segment: PeterSegment, thresholds: FilterThresholds) -> bool:
    return thresholds.min_duration_sec <= segment.duration <= thresholds.max_duration_sec


def asr_confidence_ok(segment: PeterSegment, thresholds: FilterThresholds) -> bool:
    return segment.asr_confidence >= thresholds.min_asr_confidence


def no_speech_prob_ok(segment: PeterSegment, thresholds: FilterThresholds) -> bool:
    return segment.no_speech_prob <= thresholds.max_no_speech_prob


def compression_ratio_ok(segment: PeterSegment, thresholds: FilterThresholds) -> bool:
    """Reject segments with Whisper-style repetitive token loops.

    Two signals: (a) the upstream Whisper compression_ratio metric, (b) a regex
    check for any single word repeated four-plus times in a row.
    """
    if segment.compression_ratio > thresholds.max_compression_ratio:
        return False
    return _REPEAT_RE.search(segment.text) is None


def filler_ratio(text: str) -> float:
    """Compute the fraction of tokens that are filler words or phrases."""
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    if not tokens:
        return 0.0

    filler_token_count = sum(1 for t in tokens if t in _FILLER_WORDS)

    # Count phrase occurrences in original text, weighted by phrase length in tokens.
    lowered = text.lower()
    for phrase in _FILLER_PHRASES:
        phrase_tokens = phrase.count(" ") + 1
        filler_token_count += lowered.count(phrase) * phrase_tokens

    return filler_token_count / len(tokens)


def filler_ratio_ok(segment: PeterSegment, thresholds: FilterThresholds) -> bool:
    return filler_ratio(segment.text) <= thresholds.max_filler_ratio


def overlap_ok(segment: PeterSegment, thresholds: FilterThresholds) -> bool:
    return segment.overlap_ratio <= thresholds.max_overlap_ratio


def speaker_confidence_ok(segment: PeterSegment, thresholds: FilterThresholds) -> bool:
    return segment.speaker_confidence >= thresholds.min_speaker_confidence


def topic_ok(
    similarity: float,
    thresholds: FilterThresholds,
    relative_threshold: float | None = None,
) -> bool:
    """Topic relevance gate.

    Pass if the segment's topic-similarity exceeds the absolute threshold AND, when
    provided, the relative percentile threshold (e.g., the corpus 40th percentile).
    A `None` `relative_threshold` disables relative gating; an absolute threshold of
    0.0 disables absolute gating. With both disabled the gate is always open.
    """
    if similarity < thresholds.min_topic_similarity:
        return False
    return not (relative_threshold is not None and similarity < relative_threshold)


def evaluate(
    segment: PeterSegment,
    *,
    thresholds: FilterThresholds | None = None,
    topic_similarity: float | None = None,
    relative_topic_threshold: float | None = None,
) -> FilterReport:
    """Run every filter against a segment and return the combined report.

    `topic_similarity` is the cosine between the segment text embedding and the
    Food & CPG seed-prompt embedding. Pass `None` to skip the topic gate (always pass).
    """
    t = thresholds or FilterThresholds()
    topic_pass = (
        True
        if topic_similarity is None
        else topic_ok(topic_similarity, t, relative_topic_threshold)
    )
    return FilterReport(
        duration_ok=duration_ok(segment, t),
        asr_confidence_ok=asr_confidence_ok(segment, t),
        no_speech_prob_ok=no_speech_prob_ok(segment, t),
        compression_ratio_ok=compression_ratio_ok(segment, t),
        filler_ratio_ok=filler_ratio_ok(segment, t),
        overlap_ok=overlap_ok(segment, t),
        topic_ok=topic_pass,
        speaker_confidence_ok=speaker_confidence_ok(segment, t),
    )


def percentile_threshold(similarities: list[float], percentile: float) -> float:
    """Compute the relative-similarity cutoff for the given percentile.

    `percentile` is a fraction in [0, 1]. Returns 0.0 when the list is empty.
    """
    if not similarities:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"percentile must be in [0, 1], got {percentile}")
    sorted_sims = sorted(similarities)
    idx = round(percentile * (len(sorted_sims) - 1))
    return sorted_sims[idx]

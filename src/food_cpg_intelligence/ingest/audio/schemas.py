"""Pydantic models for the audio ingestion pipeline.

All models are immutable (`frozen=True`) and use tuples instead of lists so that
pipeline stages can pass values around without worrying about hidden mutation.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

GenerationMode = Literal[
    "continuation",
    "natural_qa",
    "reverse_qa",
    "style_rewrite",
    "rag_grounded",
    "adversarial",
]


class AudioSource(BaseModel, frozen=True):
    """A downloaded audio file plus its provenance metadata."""

    source_type: Literal["youtube", "podcast", "local"]
    source_id: str
    """Stable identifier (YouTube video id, podcast guid, or local file hash)."""
    url: str = ""
    title: str = ""
    audio_path: str
    """Absolute path to a 16 kHz mono wav file ready for ASR."""
    duration_sec: float
    chapters: tuple["Chapter", ...] = ()
    sponsor_segments: tuple[tuple[float, float], ...] = ()
    """SponsorBlock-style ad / sponsor time spans, in seconds."""
    extra: dict[str, Any] = Field(default_factory=dict)


class Chapter(BaseModel, frozen=True):
    """Chapter marker from YouTube metadata or podcast show notes."""

    start: float
    end: float
    title: str


class Word(BaseModel, frozen=True):
    """A single ASR-recognised word with its alignment timestamps."""

    text: str
    start: float
    end: float
    score: float = 0.0
    """Word-level confidence from forced alignment (0..1). 0 = no alignment data."""


class TranscribedSegment(BaseModel, frozen=True):
    """A raw ASR segment before diarization is applied.

    Mirrors the faster-whisper / WhisperX output shape we actually consume.
    """

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0
    """Higher values indicate repetitive token loops, a Whisper hallucination signature."""
    language: str = "en"

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def asr_confidence(self) -> float:
        """Mean word-level alignment confidence, or 0.0 if no word data."""
        if not self.words:
            return 0.0
        return sum(w.score for w in self.words) / len(self.words)


class DiarizedTurn(BaseModel, frozen=True):
    """A diarized speaker turn produced by pyannote.audio."""

    start: float
    end: float
    speaker_label: str
    """pyannote-assigned label like 'SPEAKER_00'. Not yet identified as Peter or not."""
    overlap_ratio: float = 0.0
    """Fraction of this turn that overlaps with another speaker, in [0, 1]."""

    @property
    def duration(self) -> float:
        return self.end - self.start


class AlignedSegment(BaseModel, frozen=True):
    """A transcribed segment after diarization assignment.

    Produced by intersecting `TranscribedSegment` (text + word timing) with the
    diarized turn that best covers it. The `speaker_label` is still anonymous at
    this stage; identification happens in the speaker_id step.
    """

    start: float
    end: float
    text: str
    words: tuple[Word, ...]
    speaker_label: str
    overlap_ratio: float
    no_speech_prob: float
    compression_ratio: float
    asr_confidence: float

    @property
    def duration(self) -> float:
        return self.end - self.start


class PeterSegment(BaseModel, frozen=True):
    """An aligned segment confirmed to be Peter via speaker embedding match."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...]
    overlap_ratio: float
    no_speech_prob: float
    compression_ratio: float
    asr_confidence: float
    speaker_confidence: float
    """Cosine similarity to the Peter reference embedding (0..1)."""

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class FilterReport(BaseModel, frozen=True):
    """The pass/fail outcome of every quality filter for a segment.

    A segment is `passed` only if every filter returned True. The booleans let us
    keep per-filter rejection counters for pipeline observability.
    """

    duration_ok: bool
    asr_confidence_ok: bool
    no_speech_prob_ok: bool
    compression_ratio_ok: bool
    filler_ratio_ok: bool
    overlap_ok: bool
    topic_ok: bool
    speaker_confidence_ok: bool

    @property
    def passed(self) -> bool:
        return (
            self.duration_ok
            and self.asr_confidence_ok
            and self.no_speech_prob_ok
            and self.compression_ratio_ok
            and self.filler_ratio_ok
            and self.overlap_ok
            and self.topic_ok
            and self.speaker_confidence_ok
        )

    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(
            name for name, ok in self.model_dump().items() if name.endswith("_ok") and not ok
        )


class SFTMessage(BaseModel, frozen=True):
    role: Literal["system", "user", "assistant"]
    content: str


class SFTExample(BaseModel, frozen=True):
    """A single training example ready to be written to the SFT JSONL."""

    example_id: str
    source_id: str
    source_type: Literal["youtube", "podcast", "local"]
    source_url: str = ""
    timestamp: tuple[float, float]
    """(start, end) seconds in the source audio. Kept for traceability and future TTS."""
    speaker_confidence: float
    asr_confidence: float
    duration_sec: float
    topic_tags: tuple[str, ...] = ()
    generation_mode: GenerationMode
    scrubbed: bool
    """True if specific numbers / dates / brand names were placeholder-replaced."""
    messages: tuple[SFTMessage, ...]


class RAGChunk(BaseModel, frozen=True):
    """A retrieval chunk produced from the same audio source for sku-food ingestion."""

    chunk_id: str
    document_id: str
    """Stable per-episode id so all chunks from one episode share a parent document."""
    source_type: Literal["youtube", "podcast", "local"]
    source_url: str
    title: str
    heading: str = ""
    """Optional sub-heading (e.g., chapter title) for the chunk."""
    content: str
    timestamp: tuple[float, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


# Forward reference resolution
AudioSource.model_rebuild()

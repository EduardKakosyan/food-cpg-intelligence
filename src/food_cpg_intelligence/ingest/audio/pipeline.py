"""End-to-end audio ingestion orchestrator.

Composes downloader -> ASR -> diarization -> speaker identification -> filtering ->
JSONL output. Heavy ML resources (Transcriber, Diarizer, PeterIdentifier) are
constructed once and can be reused across episodes via `BatchProcessor`.

This module owns the alignment step (ASR segments + diarized turns -> AlignedSegment)
and the sponsor-segment removal step. Topic gating uses an embedding model that we
also lazy-import to keep the module light.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from food_cpg_intelligence.ingest.audio import filters as flt
from food_cpg_intelligence.ingest.audio.asr import AsrConfig, Transcriber
from food_cpg_intelligence.ingest.audio.diarization import DiarizationConfig, Diarizer
from food_cpg_intelligence.ingest.audio.downloader import download
from food_cpg_intelligence.ingest.audio.schemas import (
    AlignedSegment,
    AudioSource,
    DiarizedTurn,
    FilterReport,
    PeterSegment,
    RAGChunk,
    SFTExample,
    SFTMessage,
    TranscribedSegment,
)
from food_cpg_intelligence.ingest.audio.speaker_id import (
    PeterIdentifier,
    SpeakerIdConfig,
    SpeakerVerdict,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path
    asr: AsrConfig
    diarization: DiarizationConfig
    speaker_id: SpeakerIdConfig
    filters: flt.FilterThresholds
    reference_clip: Path
    hf_token: str
    skip_sponsor_segments: bool = True
    relative_topic_percentile: float | None = None
    """If set, drop segments below this percentile of topic similarity within the episode.

    Disabled by default at the per-episode level — apply globally across the whole
    corpus in a separate pass for stronger signal.
    """


@dataclass(frozen=True)
class EpisodeArtifacts:
    """Everything we record for one processed episode."""

    audio_source: AudioSource
    transcribed_segments: tuple[TranscribedSegment, ...]
    diarized_turns: tuple[DiarizedTurn, ...]
    speaker_verdict: SpeakerVerdict
    peter_segments: tuple[PeterSegment, ...]
    filter_reports: tuple[FilterReport, ...]
    sft_examples: tuple[SFTExample, ...]
    rag_chunks: tuple[RAGChunk, ...]


class BatchProcessor:
    """Holds the loaded models so they can be reused across episodes."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.transcriber = Transcriber(config.asr)
        self.diarizer = Diarizer(config.diarization, hf_token=config.hf_token)
        self.identifier = PeterIdentifier(
            config.reference_clip,
            config.speaker_id,
            device=config.diarization.device,
        )

    def process_url(self, url_or_path: str) -> EpisodeArtifacts:
        config = self.config
        audio_source = download(url_or_path, output_dir=config.output_dir / "raw")
        return self.process_source(audio_source)

    def process_source(self, audio_source: AudioSource) -> EpisodeArtifacts:
        config = self.config
        audio_path = Path(audio_source.audio_path)

        transcribed = self.transcriber.transcribe(audio_path)
        diarized = self.diarizer.diarize(audio_path)
        verdict = self.identifier.identify(audio_path, diarized)

        aligned = align_segments(transcribed, diarized)
        peter_segments = extract_peter_segments(
            aligned,
            verdict,
            audio_source.sponsor_segments if config.skip_sponsor_segments else (),
        )

        topic_similarities = _maybe_topic_score(peter_segments)
        relative_threshold = (
            flt.percentile_threshold(list(topic_similarities), config.relative_topic_percentile)
            if config.relative_topic_percentile is not None and topic_similarities
            else None
        )

        filter_reports: list[FilterReport] = []
        kept: list[PeterSegment] = []
        for seg, sim in zip(peter_segments, topic_similarities, strict=False):
            report = flt.evaluate(
                seg,
                thresholds=config.filters,
                topic_similarity=sim,
                relative_topic_threshold=relative_threshold,
            )
            filter_reports.append(report)
            if report.passed:
                kept.append(seg)

        sft_examples = tuple(_to_sft_example(audio_source, s, i) for i, s in enumerate(kept))
        rag_chunks = tuple(_to_rag_chunk(audio_source, s, i) for i, s in enumerate(kept))

        logger.info(
            "audio.pipeline.episode_done",
            source_id=audio_source.source_id,
            transcribed=len(transcribed),
            diarized=len(diarized),
            peter_raw=len(peter_segments),
            peter_kept=len(kept),
            rejection_breakdown=_rejection_breakdown(filter_reports),
        )

        return EpisodeArtifacts(
            audio_source=audio_source,
            transcribed_segments=transcribed,
            diarized_turns=diarized,
            speaker_verdict=verdict,
            peter_segments=peter_segments,
            filter_reports=tuple(filter_reports),
            sft_examples=sft_examples,
            rag_chunks=rag_chunks,
        )


def align_segments(
    transcribed: tuple[TranscribedSegment, ...],
    diarized: tuple[DiarizedTurn, ...],
) -> tuple[AlignedSegment, ...]:
    """Pair each ASR segment with the diarized turn covering it most."""
    aligned: list[AlignedSegment] = []
    for seg in transcribed:
        best_turn = _best_overlap_turn(seg, diarized)
        if best_turn is None:
            continue
        aligned.append(
            AlignedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=seg.words,
                speaker_label=best_turn.speaker_label,
                overlap_ratio=best_turn.overlap_ratio,
                no_speech_prob=seg.no_speech_prob,
                compression_ratio=seg.compression_ratio,
                asr_confidence=seg.asr_confidence,
            )
        )
    return tuple(aligned)


def _best_overlap_turn(
    seg: TranscribedSegment, turns: tuple[DiarizedTurn, ...]
) -> DiarizedTurn | None:
    best_turn: DiarizedTurn | None = None
    best_overlap = 0.0
    for turn in turns:
        i_start = max(seg.start, turn.start)
        i_end = min(seg.end, turn.end)
        overlap = max(0.0, i_end - i_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn
    return best_turn


def extract_peter_segments(
    aligned: tuple[AlignedSegment, ...],
    verdict: SpeakerVerdict,
    sponsor_segments: tuple[tuple[float, float], ...],
) -> tuple[PeterSegment, ...]:
    """Keep aligned segments whose speaker is Peter and outside any sponsor span."""
    if not verdict.matched or verdict.peter_cluster_label is None:
        logger.info("audio.pipeline.no_peter_cluster", similarities=verdict.cluster_similarities)
        return ()

    peter_label = verdict.peter_cluster_label
    confidence = verdict.cluster_confidence(peter_label)
    result: list[PeterSegment] = []
    for seg in aligned:
        if seg.speaker_label != peter_label:
            continue
        if _falls_in_any_span((seg.start, seg.end), sponsor_segments):
            continue
        result.append(
            PeterSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=seg.words,
                overlap_ratio=seg.overlap_ratio,
                no_speech_prob=seg.no_speech_prob,
                compression_ratio=seg.compression_ratio,
                asr_confidence=seg.asr_confidence,
                speaker_confidence=confidence,
            )
        )
    return tuple(result)


def _falls_in_any_span(
    interval: tuple[float, float], spans: tuple[tuple[float, float], ...]
) -> bool:
    """True if `interval` overlaps any span by at least 50% of its duration."""
    start, end = interval
    duration = end - start
    if duration <= 0:
        return False
    for s_start, s_end in spans:
        i_start = max(start, s_start)
        i_end = min(end, s_end)
        if i_end - i_start >= 0.5 * duration:
            return True
    return False


def _maybe_topic_score(segments: tuple[PeterSegment, ...]) -> tuple[float, ...]:
    """Placeholder topic similarity scorer.

    A real implementation embeds each segment with `BAAI/bge-large-en-v1.5` and
    cosines against an averaged seed-prompt embedding. We return zeros here so the
    pipeline runs end-to-end without an embedding model; wire in the real scorer
    when we add the topic gate to `BatchProcessor.__init__`.
    """
    return tuple(0.0 for _ in segments)


def _to_sft_example(source: AudioSource, segment: PeterSegment, index: int) -> SFTExample:
    """Default formatting: continuation mode with a generic eliciting user turn.

    Style-rewrite, reverse-QA, and RAG-grounded examples are generated in a later
    Claude-driven stage (see Phase 2 of the research synthesis).
    """
    return SFTExample(
        example_id=f"{source.source_id}-seg-{index:04d}",
        source_id=source.source_id,
        source_type=source.source_type,
        source_url=source.url,
        timestamp=(segment.start, segment.end),
        speaker_confidence=segment.speaker_confidence,
        asr_confidence=segment.asr_confidence,
        duration_sec=segment.duration,
        topic_tags=(),
        generation_mode="continuation",
        scrubbed=False,
        messages=(
            SFTMessage(
                role="system",
                content=(
                    "You are Peter Chapman, founder of SKUFood and a Canadian food and beverage "
                    "industry expert. Speak in first person, draw from your retail experience, "
                    "and be direct and actionable."
                ),
            ),
            SFTMessage(
                role="user",
                content="Walk me through how you think about this.",
            ),
            SFTMessage(role="assistant", content=segment.text),
        ),
    )


def _to_rag_chunk(source: AudioSource, segment: PeterSegment, index: int) -> RAGChunk:
    return RAGChunk(
        chunk_id=f"{source.source_id}-rag-{index:04d}",
        document_id=source.source_id,
        source_type=source.source_type,
        source_url=source.url,
        title=source.title,
        content=segment.text,
        timestamp=(segment.start, segment.end),
        metadata={
            "speaker_confidence": segment.speaker_confidence,
            "asr_confidence": segment.asr_confidence,
            "duration_sec": segment.duration,
        },
    )


def _rejection_breakdown(reports: list[FilterReport]) -> dict[str, int]:
    """Count rejections per filter name for observability."""
    counter: dict[str, int] = {}
    for r in reports:
        if r.passed:
            continue
        for reason in r.rejection_reasons():
            counter[reason] = counter.get(reason, 0) + 1
    return counter


def write_artifacts(artifacts: EpisodeArtifacts, *, output_dir: Path) -> None:
    """Write SFT JSONL, RAG JSONL, and a debug episode.json side by side."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_id = artifacts.audio_source.source_id

    sft_path = output_dir / f"{source_id}.sft.jsonl"
    rag_path = output_dir / f"{source_id}.rag.jsonl"
    debug_path = output_dir / f"{source_id}.episode.json"

    with sft_path.open("w", encoding="utf-8") as f:
        for ex in artifacts.sft_examples:
            f.write(ex.model_dump_json() + "\n")

    with rag_path.open("w", encoding="utf-8") as f:
        for chunk in artifacts.rag_chunks:
            f.write(chunk.model_dump_json() + "\n")

    debug_payload: dict[str, Any] = {
        "audio_source": artifacts.audio_source.model_dump(),
        "speaker_verdict": {
            "peter_cluster_label": artifacts.speaker_verdict.peter_cluster_label,
            "cluster_similarities": artifacts.speaker_verdict.cluster_similarities,
        },
        "counts": {
            "transcribed": len(artifacts.transcribed_segments),
            "diarized_turns": len(artifacts.diarized_turns),
            "peter_segments": len(artifacts.peter_segments),
            "kept": len(artifacts.sft_examples),
        },
        "rejection_breakdown": _rejection_breakdown(list(artifacts.filter_reports)),
    }
    debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    logger.info(
        "audio.pipeline.artifacts_written",
        source_id=source_id,
        sft_path=str(sft_path),
        rag_path=str(rag_path),
        debug_path=str(debug_path),
    )

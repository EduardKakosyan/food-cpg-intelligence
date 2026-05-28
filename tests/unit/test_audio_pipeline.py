"""Tests for the audio pipeline orchestration helpers (no heavy ML deps)."""

from __future__ import annotations

import pytest

from food_cpg_intelligence.ingest.audio.pipeline import (
    _best_overlap_turn,
    _falls_in_any_span,
    _rejection_breakdown,
    _to_rag_chunk,
    _to_sft_example,
    align_segments,
    extract_peter_segments,
)
from food_cpg_intelligence.ingest.audio.schemas import (
    AudioSource,
    DiarizedTurn,
    FilterReport,
    PeterSegment,
    TranscribedSegment,
)
from food_cpg_intelligence.ingest.audio.speaker_id import SpeakerVerdict


def _turn(start: float, end: float, label: str, overlap: float = 0.0) -> DiarizedTurn:
    return DiarizedTurn(start=start, end=end, speaker_label=label, overlap_ratio=overlap)


def _trans(start: float, end: float, text: str = "...") -> TranscribedSegment:
    return TranscribedSegment(start=start, end=end, text=text)


def _peter(start: float, end: float) -> PeterSegment:
    return PeterSegment(
        start=start,
        end=end,
        text="retail margins and category planning",
        words=(),
        overlap_ratio=0.0,
        no_speech_prob=0.05,
        compression_ratio=1.5,
        asr_confidence=0.85,
        speaker_confidence=0.80,
    )


def _source() -> AudioSource:
    return AudioSource(
        source_type="youtube",
        source_id="ep001",
        url="https://youtu.be/ep001",
        title="Episode 001",
        audio_path="/tmp/ep001.wav",
        duration_sec=600.0,
    )


def _report(passed: bool, reasons: tuple[str, ...] = ()) -> FilterReport:
    fields = {
        "duration_ok": True,
        "asr_confidence_ok": True,
        "no_speech_prob_ok": True,
        "compression_ratio_ok": True,
        "filler_ratio_ok": True,
        "overlap_ok": True,
        "topic_ok": True,
        "speaker_confidence_ok": True,
    }
    if not passed:
        for r in reasons:
            fields[r] = False
    return FilterReport(**fields)


class TestBestOverlapTurn:
    def test_picks_highest_overlap_turn(self) -> None:
        seg = _trans(0.0, 5.0)
        turns = (
            _turn(0.0, 1.0, "A"),
            _turn(1.5, 4.5, "B"),  # 3s overlap
            _turn(4.0, 6.0, "C"),  # 1s overlap
        )
        assert _best_overlap_turn(seg, turns).speaker_label == "B"

    def test_returns_none_when_no_overlap(self) -> None:
        seg = _trans(10.0, 11.0)
        turns = (_turn(0.0, 5.0, "A"),)
        assert _best_overlap_turn(seg, turns) is None


class TestAlignSegments:
    def test_aligned_carries_speaker_label_and_metrics(self) -> None:
        transcribed = (TranscribedSegment(start=0.0, end=2.0, text="hi", no_speech_prob=0.1),)
        diarized = (_turn(0.0, 2.0, "SPEAKER_01", overlap=0.2),)
        result = align_segments(transcribed, diarized)
        assert len(result) == 1
        assert result[0].speaker_label == "SPEAKER_01"
        assert result[0].overlap_ratio == pytest.approx(0.2)
        assert result[0].no_speech_prob == pytest.approx(0.1)

    def test_drops_segments_with_no_diarized_overlap(self) -> None:
        transcribed = (_trans(10.0, 11.0),)
        diarized = (_turn(0.0, 5.0, "A"),)
        assert align_segments(transcribed, diarized) == ()


class TestExtractPeterSegments:
    def test_empty_when_no_peter_cluster_matched(self) -> None:
        verdict = SpeakerVerdict(peter_cluster_label=None, cluster_similarities={"A": 0.4})
        aligned = align_segments((_trans(0.0, 2.0),), (_turn(0.0, 2.0, "A"),))
        assert extract_peter_segments(aligned, verdict, ()) == ()

    def test_keeps_only_peter_cluster_turns(self) -> None:
        verdict = SpeakerVerdict(
            peter_cluster_label="SPEAKER_00",
            cluster_similarities={"SPEAKER_00": 0.78, "SPEAKER_01": 0.32},
        )
        aligned = align_segments(
            (_trans(0.0, 2.0), _trans(2.0, 4.0)),
            (_turn(0.0, 2.0, "SPEAKER_00"), _turn(2.0, 4.0, "SPEAKER_01")),
        )
        peter = extract_peter_segments(aligned, verdict, ())
        assert len(peter) == 1
        assert peter[0].start == pytest.approx(0.0)
        assert peter[0].speaker_confidence == pytest.approx(0.78)

    def test_drops_segments_inside_sponsor_spans(self) -> None:
        verdict = SpeakerVerdict(
            peter_cluster_label="SPEAKER_00",
            cluster_similarities={"SPEAKER_00": 0.78},
        )
        aligned = align_segments(
            (_trans(0.0, 2.0), _trans(10.0, 12.0)),
            (_turn(0.0, 12.0, "SPEAKER_00"),),
        )
        sponsor_spans = ((9.0, 13.0),)  # covers the second segment
        peter = extract_peter_segments(aligned, verdict, sponsor_spans)
        assert len(peter) == 1
        assert peter[0].start == pytest.approx(0.0)


class TestFallsInAnySpan:
    def test_no_spans(self) -> None:
        assert _falls_in_any_span((0.0, 5.0), ()) is False

    def test_fully_inside_span(self) -> None:
        assert _falls_in_any_span((10.0, 12.0), ((9.0, 13.0),)) is True

    def test_partially_inside_below_half_threshold(self) -> None:
        # 0.5s of overlap out of 10s segment = 5% — keep.
        assert _falls_in_any_span((0.0, 10.0), ((9.5, 11.0),)) is False

    def test_majority_inside_span(self) -> None:
        # 6s of overlap out of 10s segment = 60% — drop.
        assert _falls_in_any_span((0.0, 10.0), ((4.0, 12.0),)) is True

    def test_zero_duration_segment(self) -> None:
        assert _falls_in_any_span((5.0, 5.0), ((0.0, 10.0),)) is False


class TestRejectionBreakdown:
    def test_counts_failing_filters(self) -> None:
        reports = [
            _report(True),
            _report(False, ("duration_ok",)),
            _report(False, ("duration_ok", "filler_ratio_ok")),
        ]
        counts = _rejection_breakdown(reports)
        assert counts["duration_ok"] == 2
        assert counts["filler_ratio_ok"] == 1
        assert "asr_confidence_ok" not in counts


class TestSftAndRagFormatting:
    def test_sft_example_has_three_messages(self) -> None:
        ex = _to_sft_example(_source(), _peter(0.0, 10.0), index=0)
        roles = [m.role for m in ex.messages]
        assert roles == ["system", "user", "assistant"]
        assert ex.generation_mode == "continuation"
        assert ex.source_id == "ep001"
        assert ex.example_id == "ep001-seg-0000"

    def test_rag_chunk_preserves_timestamp_and_title(self) -> None:
        chunk = _to_rag_chunk(_source(), _peter(10.0, 30.0), index=2)
        assert chunk.chunk_id == "ep001-rag-0002"
        assert chunk.document_id == "ep001"
        assert chunk.title == "Episode 001"
        assert chunk.timestamp == (10.0, 30.0)
        assert chunk.metadata["asr_confidence"] == pytest.approx(0.85)

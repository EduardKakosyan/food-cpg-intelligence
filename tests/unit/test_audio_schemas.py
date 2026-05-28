"""Tests for the audio ingestion Pydantic schemas."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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
    Word,
)


def _make_words() -> tuple[Word, ...]:
    return (
        Word(text="hello", start=0.0, end=0.5, score=0.9),
        Word(text="world", start=0.5, end=1.0, score=0.8),
    )


class TestWordAndSegment:
    def test_word_is_frozen(self) -> None:
        w = Word(text="hi", start=0.0, end=0.1)
        with pytest.raises(ValidationError):
            w.text = "bye"  # type: ignore[misc]

    def test_transcribed_segment_duration(self) -> None:
        seg = TranscribedSegment(start=1.0, end=4.5, text="hello world", words=_make_words())
        assert seg.duration == pytest.approx(3.5)

    def test_transcribed_segment_asr_confidence_average(self) -> None:
        seg = TranscribedSegment(start=0.0, end=1.0, text="hello world", words=_make_words())
        assert seg.asr_confidence == pytest.approx(0.85)

    def test_transcribed_segment_asr_confidence_zero_without_words(self) -> None:
        seg = TranscribedSegment(start=0.0, end=1.0, text="text", words=())
        assert seg.asr_confidence == 0.0


class TestAudioSource:
    def test_round_trip_json(self) -> None:
        src = AudioSource(
            source_type="youtube",
            source_id="abc123",
            url="https://youtu.be/abc123",
            title="ep 1",
            audio_path="/tmp/abc123.wav",
            duration_sec=1234.5,
        )
        payload = src.model_dump_json()
        restored = AudioSource.model_validate(json.loads(payload))
        assert restored == src


class TestPeterSegment:
    def test_duration_and_word_count(self) -> None:
        seg = PeterSegment(
            start=10.0,
            end=14.0,
            text="we ship products to canadian retailers",
            words=(),
            overlap_ratio=0.0,
            no_speech_prob=0.05,
            compression_ratio=1.5,
            asr_confidence=0.85,
            speaker_confidence=0.78,
        )
        assert seg.duration == pytest.approx(4.0)
        assert seg.word_count == 6


class TestFilterReport:
    def _all_pass(self) -> FilterReport:
        return FilterReport(
            duration_ok=True,
            asr_confidence_ok=True,
            no_speech_prob_ok=True,
            compression_ratio_ok=True,
            filler_ratio_ok=True,
            overlap_ok=True,
            topic_ok=True,
            speaker_confidence_ok=True,
        )

    def test_passed_true_when_all_filters_pass(self) -> None:
        assert self._all_pass().passed is True

    def test_passed_false_if_any_filter_fails(self) -> None:
        r = self._all_pass().model_copy(update={"overlap_ok": False})
        assert r.passed is False

    def test_rejection_reasons_lists_failing_filters(self) -> None:
        r = self._all_pass().model_copy(update={"overlap_ok": False, "filler_ratio_ok": False})
        reasons = set(r.rejection_reasons())
        assert reasons == {"overlap_ok", "filler_ratio_ok"}


class TestSFTExample:
    def test_round_trip_json(self) -> None:
        example = SFTExample(
            example_id="ep1-seg-0001",
            source_id="ep1",
            source_type="youtube",
            source_url="https://youtu.be/ep1",
            timestamp=(10.0, 42.5),
            speaker_confidence=0.8,
            asr_confidence=0.9,
            duration_sec=32.5,
            generation_mode="continuation",
            scrubbed=False,
            messages=(
                SFTMessage(role="system", content="You are Peter."),
                SFTMessage(role="user", content="Tell me."),
                SFTMessage(role="assistant", content="Here's what I think."),
            ),
        )
        payload = example.model_dump_json()
        restored = SFTExample.model_validate(json.loads(payload))
        assert restored == example
        # JSONL: assert one example serialises to a single line of valid JSON.
        assert "\n" not in payload


class TestRAGChunk:
    def test_metadata_preserved(self) -> None:
        chunk = RAGChunk(
            chunk_id="ep1-rag-0001",
            document_id="ep1",
            source_type="podcast",
            source_url="https://example.com/ep1.mp3",
            title="ep 1",
            content="hello",
            timestamp=(0.0, 10.0),
            metadata={"asr_confidence": 0.91},
        )
        assert chunk.metadata["asr_confidence"] == pytest.approx(0.91)


class TestAlignedAndDiarized:
    def test_aligned_segment_duration(self) -> None:
        seg = AlignedSegment(
            start=2.0,
            end=7.0,
            text="...",
            words=(),
            speaker_label="SPEAKER_00",
            overlap_ratio=0.0,
            no_speech_prob=0.05,
            compression_ratio=1.2,
            asr_confidence=0.7,
        )
        assert seg.duration == pytest.approx(5.0)

    def test_diarized_turn_clamps_make_sense(self) -> None:
        # Overlap ratios are produced by our own helper, so just check the field accepts [0,1].
        DiarizedTurn(start=0.0, end=1.0, speaker_label="X", overlap_ratio=0.0)
        DiarizedTurn(start=0.0, end=1.0, speaker_label="X", overlap_ratio=1.0)

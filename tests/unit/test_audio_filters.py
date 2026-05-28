"""Tests for the audio quality filters."""

from __future__ import annotations

import pytest

from food_cpg_intelligence.ingest.audio import filters as flt
from food_cpg_intelligence.ingest.audio.schemas import PeterSegment, Word


def _seg(
    *,
    start: float = 0.0,
    end: float = 10.0,
    text: str = "this is a long enough sentence about canadian retailers and shelf strategy.",
    words: tuple[Word, ...] = (),
    overlap_ratio: float = 0.0,
    no_speech_prob: float = 0.05,
    compression_ratio: float = 1.5,
    asr_confidence: float = 0.85,
    speaker_confidence: float = 0.80,
) -> PeterSegment:
    return PeterSegment(
        start=start,
        end=end,
        text=text,
        words=words,
        overlap_ratio=overlap_ratio,
        no_speech_prob=no_speech_prob,
        compression_ratio=compression_ratio,
        asr_confidence=asr_confidence,
        speaker_confidence=speaker_confidence,
    )


T = flt.FilterThresholds()


class TestDuration:
    def test_too_short_fails(self) -> None:
        assert flt.duration_ok(_seg(start=0.0, end=2.0), T) is False

    def test_too_long_fails(self) -> None:
        assert flt.duration_ok(_seg(start=0.0, end=120.0), T) is False

    def test_in_range_passes(self) -> None:
        assert flt.duration_ok(_seg(start=0.0, end=30.0), T) is True


class TestAsrConfidence:
    def test_low_confidence_fails(self) -> None:
        assert flt.asr_confidence_ok(_seg(asr_confidence=0.4), T) is False

    def test_at_threshold_passes(self) -> None:
        assert flt.asr_confidence_ok(_seg(asr_confidence=0.6), T) is True


class TestNoSpeechProb:
    def test_high_no_speech_prob_fails(self) -> None:
        assert flt.no_speech_prob_ok(_seg(no_speech_prob=0.7), T) is False

    def test_low_passes(self) -> None:
        assert flt.no_speech_prob_ok(_seg(no_speech_prob=0.05), T) is True


class TestCompressionRatio:
    def test_high_ratio_fails(self) -> None:
        assert flt.compression_ratio_ok(_seg(compression_ratio=3.0), T) is False

    def test_repeating_word_loop_fails(self) -> None:
        # 4 repeats triggers the regex.
        seg = _seg(text="yeah yeah yeah yeah and that's how the market works")
        assert flt.compression_ratio_ok(seg, T) is False

    def test_clean_text_passes(self) -> None:
        assert flt.compression_ratio_ok(_seg(text="this is a normal sentence"), T) is True


class TestFillerRatio:
    def test_pure_filler_is_high(self) -> None:
        assert flt.filler_ratio("um uh um uh er") == pytest.approx(1.0)

    def test_clean_text_is_zero(self) -> None:
        assert flt.filler_ratio("retail margin and category management") == 0.0

    def test_phrase_filler_counted_with_token_weight(self) -> None:
        # 10 word tokens total. "you know" appears 2x and contributes 2 phrase-tokens
        # each, so filler count = 4 / 10 = 0.4.
        ratio = flt.filler_ratio("we go you know we go you know we go")
        assert ratio == pytest.approx(0.4)

    def test_empty_text(self) -> None:
        assert flt.filler_ratio("") == 0.0

    def test_threshold_gate_passes(self) -> None:
        assert (
            flt.filler_ratio_ok(_seg(text="retail margin and category strategy work."), T) is True
        )

    def test_threshold_gate_fails_high_filler(self) -> None:
        assert flt.filler_ratio_ok(_seg(text="um uh um um uh er"), T) is False


class TestOverlap:
    def test_high_overlap_fails(self) -> None:
        assert flt.overlap_ok(_seg(overlap_ratio=0.5), T) is False

    def test_low_overlap_passes(self) -> None:
        assert flt.overlap_ok(_seg(overlap_ratio=0.05), T) is True


class TestSpeakerConfidence:
    def test_low_speaker_confidence_fails(self) -> None:
        assert flt.speaker_confidence_ok(_seg(speaker_confidence=0.3), T) is False

    def test_at_threshold_passes(self) -> None:
        assert flt.speaker_confidence_ok(_seg(speaker_confidence=0.55), T) is True


class TestTopicGate:
    def test_absolute_threshold_below_fails(self) -> None:
        thresholds = flt.FilterThresholds(min_topic_similarity=0.5)
        assert flt.topic_ok(0.3, thresholds) is False

    def test_absolute_threshold_disabled(self) -> None:
        thresholds = flt.FilterThresholds(min_topic_similarity=0.0)
        assert flt.topic_ok(0.3, thresholds) is True

    def test_relative_threshold_below_fails(self) -> None:
        assert flt.topic_ok(0.4, T, relative_threshold=0.6) is False

    def test_relative_threshold_above_passes(self) -> None:
        assert flt.topic_ok(0.7, T, relative_threshold=0.6) is True

    def test_none_relative_disables(self) -> None:
        assert flt.topic_ok(0.0, T, relative_threshold=None) is True


class TestPercentileThreshold:
    def test_empty_returns_zero(self) -> None:
        assert flt.percentile_threshold([], 0.5) == 0.0

    def test_percentile_in_middle(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        # p50 of 5-element list at idx round(0.5 * 4) = 2 -> value 0.3
        assert flt.percentile_threshold(values, 0.5) == pytest.approx(0.3)

    def test_percentile_at_top(self) -> None:
        assert flt.percentile_threshold([0.1, 0.5, 0.9], 1.0) == pytest.approx(0.9)

    def test_invalid_percentile_raises(self) -> None:
        with pytest.raises(ValueError):
            flt.percentile_threshold([0.1, 0.2], 1.5)


class TestEvaluate:
    def test_all_pass_for_clean_segment(self) -> None:
        report = flt.evaluate(_seg(text="retail margin and category management"))
        assert report.passed is True
        assert report.rejection_reasons() == ()

    def test_reports_all_failures(self) -> None:
        bad = _seg(
            start=0.0,
            end=1.0,  # fails duration
            text="um uh um uh",  # fails filler
            asr_confidence=0.2,  # fails ASR
            speaker_confidence=0.1,  # fails speaker
        )
        report = flt.evaluate(bad)
        reasons = set(report.rejection_reasons())
        assert "duration_ok" in reasons
        assert "asr_confidence_ok" in reasons
        assert "filler_ratio_ok" in reasons
        assert "speaker_confidence_ok" in reasons

    def test_topic_gate_can_fail_otherwise_clean_segment(self) -> None:
        report = flt.evaluate(
            _seg(text="retail margin"),
            thresholds=flt.FilterThresholds(min_topic_similarity=0.5),
            topic_similarity=0.3,
        )
        assert report.topic_ok is False
        assert report.passed is False

    def test_topic_gate_disabled_when_similarity_none(self) -> None:
        report = flt.evaluate(_seg(text="retail margin"), topic_similarity=None)
        assert report.topic_ok is True

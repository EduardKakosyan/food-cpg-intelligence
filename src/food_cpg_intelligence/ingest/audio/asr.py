"""ASR + forced alignment using WhisperX (faster-whisper backend).

The `Transcriber` class loads the Whisper model and the wav2vec2 alignment model
once and can be reused across episodes. Heavy deps are imported lazily on construction
so importing this module is cheap.

Device matrix:
- CUDA: float16 ASR, float32 alignment (the WhisperX default)
- CPU:  int8  ASR, float32 alignment
- MPS:  not supported by faster-whisper (CTranslate2 backend has no MPS path); fall
        back to CPU. Apple Silicon users should expect CPU-only ASR locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from food_cpg_intelligence.ingest.audio.schemas import TranscribedSegment, Word

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AsrConfig:
    model: str = "large-v3-turbo"
    compute_type: str = "auto"
    device: str = "auto"
    language: str = "en"
    batch_size: int = 16
    align_model: str | None = None


def _detect_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("torch required for ASR device detection.") from exc
    if torch.cuda.is_available():
        return "cuda"
    # CTranslate2 backs faster-whisper and has no MPS path. Always fall to CPU.
    return "cpu"


def _detect_compute_type(device: str, preferred: str) -> str:
    if preferred != "auto":
        return preferred
    return "float16" if device == "cuda" else "int8"


def _clamp_batch_size(batch_size: int, device: str) -> int:
    return batch_size if device == "cuda" else min(batch_size, 4)


class Transcriber:
    """Loads Whisper + wav2vec2 alignment once and transcribes audio files."""

    def __init__(self, config: AsrConfig | None = None) -> None:
        self.config = config or AsrConfig()
        self.device = _detect_device(self.config.device)
        self.compute_type = _detect_compute_type(self.device, self.config.compute_type)
        self.batch_size = _clamp_batch_size(self.config.batch_size, self.device)
        logger.info(
            "audio.asr.init",
            model=self.config.model,
            device=self.device,
            compute_type=self.compute_type,
            batch_size=self.batch_size,
        )
        self._whisperx = self._load_whisperx()
        self._asr_model: Any = self._whisperx.load_model(
            self.config.model,
            self.device,
            compute_type=self.compute_type,
            language=self.config.language,
        )
        self._align_model: Any | None = None
        self._align_metadata: Any | None = None

    @staticmethod
    def _load_whisperx() -> Any:
        try:
            import whisperx
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "whisperx not installed. Install the audio extra: `uv sync --extra audio`."
            ) from exc
        return whisperx

    def _ensure_align_model(self, language: str) -> None:
        if self._align_model is not None:
            return
        self._align_model, self._align_metadata = self._whisperx.load_align_model(
            language_code=language,
            device=self.device,
            model_name=self.config.align_model,
        )

    def transcribe(self, audio_path: Path) -> tuple[TranscribedSegment, ...]:
        """Transcribe an audio file and return word-aligned segments."""
        audio = self._whisperx.load_audio(str(audio_path))

        logger.info("audio.asr.transcribe.start", path=str(audio_path))
        result = self._asr_model.transcribe(audio, batch_size=self.batch_size)
        language = result.get("language", self.config.language)

        self._ensure_align_model(language)
        aligned = self._whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        segments = tuple(_to_transcribed_segment(s, language) for s in aligned["segments"])
        logger.info(
            "audio.asr.transcribe.done",
            path=str(audio_path),
            segments=len(segments),
            language=language,
        )
        return segments


def _to_transcribed_segment(raw: dict[str, Any], language: str) -> TranscribedSegment:
    """Convert a WhisperX aligned segment dict into our Pydantic schema."""
    words = tuple(
        Word(
            text=str(w.get("word", "")).strip(),
            start=float(w.get("start", raw.get("start", 0.0))),
            end=float(w.get("end", raw.get("end", 0.0))),
            score=float(w.get("score", 0.0)),
        )
        for w in raw.get("words", [])
        if w.get("word")
    )
    return TranscribedSegment(
        start=float(raw["start"]),
        end=float(raw["end"]),
        text=str(raw.get("text", "")).strip(),
        words=words,
        no_speech_prob=float(raw.get("no_speech_prob", 0.0)),
        compression_ratio=float(raw.get("compression_ratio", 0.0)),
        language=language,
    )

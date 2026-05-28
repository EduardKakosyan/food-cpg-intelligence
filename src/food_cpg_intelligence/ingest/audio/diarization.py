"""Speaker diarization using pyannote.audio 3.1.

pyannote 3.1 is a gated model on Hugging Face. Set `FCPG_HUGGINGFACE_TOKEN` (or pass
`hf_token` explicitly) after accepting the model license on hf.co/pyannote/speaker-diarization-3.1.

On Apple Silicon, force `device='cpu'` — pyannote has no MPS path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from food_cpg_intelligence.ingest.audio.schemas import DiarizedTurn

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DiarizationConfig:
    model: str = "pyannote/speaker-diarization-3.1"
    device: str = "cpu"
    min_speakers: int | None = None
    max_speakers: int | None = None


class Diarizer:
    """Loads the pyannote pipeline once and reuses it across episodes."""

    def __init__(self, config: DiarizationConfig | None = None, *, hf_token: str = "") -> None:
        self.config = config or DiarizationConfig()
        if not hf_token:
            raise ValueError(
                "A Hugging Face token is required for pyannote. Set FCPG_HUGGINGFACE_TOKEN "
                "after accepting the license at hf.co/pyannote/speaker-diarization-3.1."
            )
        self._pipeline = self._load_pipeline(self.config.model, hf_token, self.config.device)
        logger.info(
            "audio.diarization.init",
            model=self.config.model,
            device=self.config.device,
        )

    @staticmethod
    def _load_pipeline(model: str, hf_token: str, device: str) -> Any:
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pyannote.audio not installed. Install the audio extra: " "`uv sync --extra audio`."
            ) from exc
        pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        return pipeline

    def diarize(self, audio_path: Path) -> tuple[DiarizedTurn, ...]:
        """Run diarization and return turns with overlap ratios computed."""
        logger.info("audio.diarization.start", path=str(audio_path))
        kwargs: dict[str, int] = {}
        if self.config.min_speakers is not None:
            kwargs["min_speakers"] = self.config.min_speakers
        if self.config.max_speakers is not None:
            kwargs["max_speakers"] = self.config.max_speakers

        diarization = self._pipeline(str(audio_path), **kwargs)

        # Raw turns from pyannote — labels are anonymous (SPEAKER_00, SPEAKER_01, ...)
        raw_turns: list[tuple[float, float, str]] = [
            (float(turn.start), float(turn.end), str(speaker))
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # Compute overlap_ratio per turn against all other-speaker turns.
        turns = tuple(
            DiarizedTurn(
                start=start,
                end=end,
                speaker_label=speaker,
                overlap_ratio=_overlap_ratio((start, end, speaker), raw_turns),
            )
            for start, end, speaker in raw_turns
        )
        logger.info(
            "audio.diarization.done",
            path=str(audio_path),
            turns=len(turns),
            speakers=len({t.speaker_label for t in turns}),
        )
        return turns


def _overlap_ratio(
    turn: tuple[float, float, str],
    all_turns: list[tuple[float, float, str]],
) -> float:
    """Fraction of `turn` that intersects any other speaker's turn, in [0, 1]."""
    start, end, speaker = turn
    duration = end - start
    if duration <= 0:
        return 0.0
    overlap = 0.0
    for o_start, o_end, o_speaker in all_turns:
        if o_speaker == speaker:
            continue
        i_start = max(start, o_start)
        i_end = min(end, o_end)
        if i_end > i_start:
            overlap += i_end - i_start
    return min(1.0, overlap / duration)

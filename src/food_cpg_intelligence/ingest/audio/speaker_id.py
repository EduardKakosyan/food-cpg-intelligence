"""Speaker identification: match diarization clusters against a Peter reference clip.

Uses ECAPA-TDNN embeddings (SpeechBrain) at 16 kHz. For each diarized cluster we
encode its longest turns, average them into a centroid, and cosine-compare against
the reference embedding. The cluster with the highest similarity above
`cosine_threshold` is labelled Peter; if no cluster crosses the threshold, no
turns are accepted as Peter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from food_cpg_intelligence.ingest.audio.schemas import DiarizedTurn

logger = structlog.get_logger(__name__)

_TARGET_SR = 16_000


@dataclass(frozen=True)
class SpeakerIdConfig:
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    cosine_threshold: float = 0.55
    min_clip_seconds_for_centroid: float = 3.0
    max_clips_per_cluster: int = 8
    """Cap per-cluster turns used for the centroid to keep compute bounded on long episodes."""
    cache_dir: str = "models/audio_cache/ecapa"


@dataclass(frozen=True)
class SpeakerVerdict:
    """The output of identifying Peter among diarized clusters."""

    peter_cluster_label: str | None
    """The pyannote cluster label assigned to Peter, or None if no cluster matched."""
    cluster_similarities: dict[str, float]
    """All cluster -> cosine similarity scores for observability."""

    @property
    def matched(self) -> bool:
        return self.peter_cluster_label is not None

    def cluster_confidence(self, label: str) -> float:
        return self.cluster_similarities.get(label, 0.0)


class PeterIdentifier:
    """Encodes the Peter reference once and matches diarization clusters against it."""

    def __init__(
        self,
        reference_clip: Path,
        config: SpeakerIdConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        self.config = config or SpeakerIdConfig()
        self.device = device
        if not reference_clip.exists():
            raise FileNotFoundError(
                f"Peter reference clip not found at {reference_clip}. "
                "Prepare a 30-60s clean recording of Peter speaking only."
            )
        self._classifier = self._load_classifier(
            self.config.embedding_model, self.config.cache_dir, device
        )
        self._reference_embedding = self._encode_file(reference_clip)
        logger.info(
            "audio.speaker_id.init",
            reference=str(reference_clip),
            embedding_model=self.config.embedding_model,
            cosine_threshold=self.config.cosine_threshold,
        )

    @staticmethod
    def _load_classifier(model_name: str, cache_dir: str, device: str) -> Any:
        try:
            from speechbrain.inference.speaker import (
                EncoderClassifier,
            )
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "speechbrain not installed. Install the audio extra: " "`uv sync --extra audio`."
            ) from exc
        return EncoderClassifier.from_hparams(
            source=model_name,
            savedir=cache_dir,
            run_opts={"device": device},
        )

    def identify(
        self,
        audio_path: Path,
        turns: tuple[DiarizedTurn, ...],
    ) -> SpeakerVerdict:
        """Return which diarized cluster (if any) is Peter, with all similarities."""
        clusters = _group_by_cluster(turns)
        similarities: dict[str, float] = {}
        for label, cluster_turns in clusters.items():
            centroid = self._cluster_centroid(audio_path, cluster_turns)
            if centroid is None:
                similarities[label] = 0.0
                continue
            similarities[label] = float(self._cosine(centroid, self._reference_embedding))

        if not similarities:
            return SpeakerVerdict(peter_cluster_label=None, cluster_similarities={})

        best_label, best_sim = max(similarities.items(), key=lambda kv: kv[1])
        matched = best_sim >= self.config.cosine_threshold
        logger.info(
            "audio.speaker_id.identify",
            path=str(audio_path),
            best_label=best_label,
            best_similarity=best_sim,
            matched=matched,
            threshold=self.config.cosine_threshold,
            n_clusters=len(similarities),
        )
        return SpeakerVerdict(
            peter_cluster_label=best_label if matched else None,
            cluster_similarities=similarities,
        )

    def _cluster_centroid(
        self,
        audio_path: Path,
        cluster_turns: list[DiarizedTurn],
    ) -> Any | None:
        """Encode up to `max_clips_per_cluster` long turns and return normalised mean."""
        long_turns = sorted(
            (t for t in cluster_turns if t.duration >= self.config.min_clip_seconds_for_centroid),
            key=lambda t: t.duration,
            reverse=True,
        )[: self.config.max_clips_per_cluster]
        if not long_turns:
            return None

        embeddings = [self._encode_segment(audio_path, t.start, t.end) for t in long_turns]
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch required for speaker_id.") from exc
        stacked = torch.stack(embeddings, dim=0)
        centroid = stacked.mean(dim=0)
        return centroid / (centroid.norm() + 1e-8)

    def _encode_file(self, path: Path) -> Any:
        """Encode an entire audio file and return a unit-norm embedding."""
        waveform = _load_wav_mono16k(path)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch required for speaker_id.") from exc
        wav = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        emb = self._classifier.encode_batch(wav).squeeze().detach()
        return emb / (emb.norm() + 1e-8)

    def _encode_segment(self, audio_path: Path, start: float, end: float) -> Any:
        """Encode a time-slice of an audio file."""
        waveform = _load_wav_mono16k(audio_path, start=start, end=end)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch required for speaker_id.") from exc
        wav = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        emb = self._classifier.encode_batch(wav).squeeze().detach()
        return emb / (emb.norm() + 1e-8)

    @staticmethod
    def _cosine(a: Any, b: Any) -> float:
        """Cosine of two unit-norm tensors. (Inner product since both are unit norm.)"""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch required for speaker_id.") from exc
        return float(torch.dot(a.flatten(), b.flatten()).item())


def _group_by_cluster(turns: tuple[DiarizedTurn, ...]) -> dict[str, list[DiarizedTurn]]:
    clusters: dict[str, list[DiarizedTurn]] = {}
    for t in turns:
        clusters.setdefault(t.speaker_label, []).append(t)
    return clusters


def _load_wav_mono16k(path: Path, *, start: float = 0.0, end: float | None = None) -> Any:
    """Load (a slice of) a wav file and return a float32 numpy array at 16 kHz mono.

    The pipeline always produces 16 kHz mono wav up-front (see downloader), so we do
    not resample here — we just slice and assert.
    """
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile / numpy required. Install the audio extra: `uv sync --extra audio`."
        ) from exc

    with sf.SoundFile(str(path)) as f:
        if f.samplerate != _TARGET_SR:
            raise ValueError(
                f"Expected 16 kHz mono wav (downloader output), got {f.samplerate} Hz "
                f"at {path}. Re-run download to normalise."
            )
        start_frame = int(start * _TARGET_SR)
        f.seek(start_frame)
        if end is None:
            data = f.read(dtype="float32", always_2d=False)
        else:
            num_frames = int((end - start) * _TARGET_SR)
            data = f.read(frames=num_frames, dtype="float32", always_2d=False)

    if data.ndim == 2:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype="float32")

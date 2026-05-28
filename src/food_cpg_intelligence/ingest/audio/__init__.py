"""Audio ingestion pipeline: podcasts/YouTube -> diarized Peter-only segments.

Stages: download -> ASR + word alignment -> diarization -> speaker identification ->
quality filtering -> SFT/RAG output JSONL.

Heavy ML dependencies (whisperx, pyannote.audio, speechbrain) are lazy-imported from
the stage modules so that the package can be imported in environments without GPU
toolchains installed.
"""

from food_cpg_intelligence.ingest.audio.schemas import (
    AlignedSegment,
    AudioSource,
    DiarizedTurn,
    FilterReport,
    PeterSegment,
    RAGChunk,
    SFTExample,
    TranscribedSegment,
    Word,
)

__all__ = [
    "AlignedSegment",
    "AudioSource",
    "DiarizedTurn",
    "FilterReport",
    "PeterSegment",
    "RAGChunk",
    "SFTExample",
    "TranscribedSegment",
    "Word",
]

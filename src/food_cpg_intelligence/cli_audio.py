"""CLI subcommands for audio ingestion (podcasts and YouTube)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml

from food_cpg_intelligence.config import settings
from food_cpg_intelligence.logging import configure_logging

app = typer.Typer(name="audio", help="Podcast / YouTube ingestion -> SFT + RAG JSONL.")


def _load_audio_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"Audio config not found at {path}")
    with path.open() as f:
        loaded: dict[str, Any] = yaml.safe_load(f) or {}
    return loaded


def _build_pipeline_config(cfg: dict[str, Any], output_dir: Path) -> Any:
    """Construct the PipelineConfig from the YAML dict + global settings.

    Lazy-imports the heavy pipeline so that `fcpg --help` and unrelated subcommands
    don't pay the cost of loading torch / whisperx / pyannote.
    """
    from food_cpg_intelligence.ingest.audio import filters as flt
    from food_cpg_intelligence.ingest.audio.asr import AsrConfig
    from food_cpg_intelligence.ingest.audio.diarization import DiarizationConfig
    from food_cpg_intelligence.ingest.audio.pipeline import PipelineConfig
    from food_cpg_intelligence.ingest.audio.speaker_id import SpeakerIdConfig

    asr_cfg = AsrConfig(**cfg.get("asr", {}))
    diar_cfg = DiarizationConfig(**cfg.get("diarization", {}))
    spk_cfg = SpeakerIdConfig(
        **{k: v for k, v in cfg.get("speaker_id", {}).items() if k != "reference_clip"}
    )
    filters_cfg_dict = cfg.get("filters", {})
    relative_pct = filters_cfg_dict.pop("relative_topic_percentile", None)
    filters_cfg = flt.FilterThresholds(**filters_cfg_dict)
    reference_clip = Path(
        cfg.get("speaker_id", {}).get("reference_clip", settings.peter_reference_clip)
    )

    if not settings.huggingface_token:
        raise typer.BadParameter(
            "FCPG_HUGGINGFACE_TOKEN not set. Required for pyannote diarization."
        )

    return PipelineConfig(
        output_dir=output_dir,
        asr=asr_cfg,
        diarization=diar_cfg,
        speaker_id=spk_cfg,
        filters=filters_cfg,
        reference_clip=reference_clip,
        hf_token=settings.huggingface_token,
        relative_topic_percentile=relative_pct,
    )


@app.command()
def process_url(
    url: str = typer.Argument(..., help="YouTube URL, podcast audio URL, or local audio path."),
    config: str = typer.Option("", help="Path to audio config YAML."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Run the full ingestion pipeline on a single URL and write SFT + RAG JSONL."""
    configure_logging()
    cfg_path = Path(config) if config else settings.resolve_path(settings.audio_config_path)
    out_path = (
        Path(output_dir) if output_dir else settings.resolve_path(settings.audio_processed_dir)
    )
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = _load_audio_config(cfg_path)
    pipeline_config = _build_pipeline_config(cfg, output_dir=out_path)

    from food_cpg_intelligence.ingest.audio.pipeline import BatchProcessor, write_artifacts

    processor = BatchProcessor(pipeline_config)
    artifacts = processor.process_url(url)
    write_artifacts(artifacts, output_dir=out_path)

    typer.echo(
        f"Done: {artifacts.audio_source.source_id} -> "
        f"{len(artifacts.sft_examples)} SFT examples, "
        f"{len(artifacts.rag_chunks)} RAG chunks at {out_path}"
    )


@app.command()
def process_batch(
    urls_file: str = typer.Argument(..., help="Newline-delimited file of URLs / local paths."),
    config: str = typer.Option("", help="Path to audio config YAML."),
    output_dir: str = typer.Option("", help="Path to output directory."),
) -> None:
    """Process every URL in `urls_file`, sharing one loaded set of models."""
    configure_logging()
    urls_path = Path(urls_file)
    if not urls_path.exists():
        raise typer.BadParameter(f"URLs file not found at {urls_path}")
    urls = [line.strip() for line in urls_path.read_text().splitlines() if line.strip()]
    if not urls:
        typer.echo("No URLs found.")
        raise typer.Exit(code=0)

    cfg_path = Path(config) if config else settings.resolve_path(settings.audio_config_path)
    out_path = (
        Path(output_dir) if output_dir else settings.resolve_path(settings.audio_processed_dir)
    )
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = _load_audio_config(cfg_path)
    pipeline_config = _build_pipeline_config(cfg, output_dir=out_path)

    from food_cpg_intelligence.ingest.audio.pipeline import BatchProcessor, write_artifacts

    processor = BatchProcessor(pipeline_config)
    total_sft = 0
    total_rag = 0
    for i, url in enumerate(urls, start=1):
        typer.echo(f"[{i}/{len(urls)}] processing {url}")
        artifacts = processor.process_url(url)
        write_artifacts(artifacts, output_dir=out_path)
        total_sft += len(artifacts.sft_examples)
        total_rag += len(artifacts.rag_chunks)
    typer.echo(
        f"Done: {len(urls)} sources -> {total_sft} SFT examples, {total_rag} RAG chunks at {out_path}"
    )


@app.command()
def verify_reference(
    reference_clip: str = typer.Option(
        "",
        help="Path to Peter reference clip; defaults to FCPG_PETER_REFERENCE_CLIP / settings.",
    ),
) -> None:
    """Check that the reference clip exists, is 16 kHz mono, and is non-empty."""
    configure_logging()
    path = Path(reference_clip) if reference_clip else Path(settings.peter_reference_clip)
    if not path.exists():
        typer.echo(f"ERROR: reference clip not found at {path}")
        raise typer.Exit(code=1)

    try:
        import soundfile as sf
    except ImportError:
        typer.echo("ERROR: soundfile not installed. Run `uv sync --extra audio`.")
        raise typer.Exit(code=1) from None

    info = sf.info(str(path))
    duration = info.frames / info.samplerate
    ok = info.samplerate == 16_000 and info.channels == 1 and duration >= 10.0
    typer.echo(f"Path:        {path}")
    typer.echo(
        f"Sample rate: {info.samplerate} Hz {'OK' if info.samplerate == 16_000 else 'EXPECTED 16000'}"
    )
    typer.echo(
        f"Channels:    {info.channels} {'OK' if info.channels == 1 else 'EXPECTED 1 (mono)'}"
    )
    typer.echo(f"Duration:    {duration:.1f}s {'OK' if duration >= 10.0 else 'EXPECTED >= 30s'}")
    if not ok:
        typer.echo(
            "\nFix with: ffmpeg -i input.mp3 -ac 1 -ar 16000 -vn -f wav data/raw/audio/peter_reference.wav"
        )
        raise typer.Exit(code=1)
    typer.echo("\nReady to use as speaker-id reference.")

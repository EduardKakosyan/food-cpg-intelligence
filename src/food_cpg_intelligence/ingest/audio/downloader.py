"""Audio download stage: YouTube and podcast RSS -> 16 kHz mono wav.

Heavy deps (yt-dlp, ffmpeg-python, feedparser) are lazy-imported so that this
module loads cleanly in environments where the `audio` extra isn't installed.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import structlog

from food_cpg_intelligence.ingest.audio.schemas import AudioSource, Chapter

logger = structlog.get_logger(__name__)

_YOUTUBE_RE = re.compile(r"(?:youtube\.com|youtu\.be)", re.IGNORECASE)


def _stable_id(url: str) -> str:
    """Deterministic short id derived from the URL hash, used when the source
    doesn't expose its own stable identifier."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg binary not found on PATH. Install via `brew install ffmpeg` (macOS) "
            "or the platform equivalent."
        )


def _convert_to_pcm_wav(src: Path, dst: Path) -> None:
    """Re-encode `src` to 16 kHz mono PCM wav at `dst`, the format WhisperX expects."""
    _ensure_ffmpeg()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        "-f",
        "wav",
        str(dst),
    ]
    logger.info("audio.downloader.ffmpeg", src=str(src), dst=str(dst))
    subprocess.run(cmd, check=True, capture_output=True)


def download_youtube(
    url: str,
    *,
    output_dir: Path,
    fetch_sponsor_segments: bool = True,
) -> AudioSource:
    """Download a YouTube video's audio track and convert to 16 kHz mono wav."""
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "yt-dlp not installed. Install the audio extra: `uv sync --extra audio`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "writethumbnail": False,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extract_flat": False,
    }
    if fetch_sponsor_segments:
        ydl_opts["postprocessors"] = [
            {
                "key": "SponsorBlock",
                "categories": ["sponsor", "selfpromo", "interaction"],
                "when": "after_filter",
            }
        ]

    logger.info("audio.downloader.youtube.start", url=url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    video_id: str = info["id"]
    downloaded_path = Path(info["requested_downloads"][0]["filepath"])
    wav_path = output_dir / f"{video_id}.wav"
    _convert_to_pcm_wav(downloaded_path, wav_path)

    chapters = tuple(
        Chapter(start=float(c["start_time"]), end=float(c["end_time"]), title=c.get("title", ""))
        for c in (info.get("chapters") or [])
    )
    sponsor_segments = tuple(
        (float(seg["segment"][0]), float(seg["segment"][1]))
        for seg in (info.get("sponsorblock_chapters") or [])
    )

    logger.info(
        "audio.downloader.youtube.done",
        url=url,
        video_id=video_id,
        duration_sec=info.get("duration", 0.0),
        chapters=len(chapters),
        sponsor_segments=len(sponsor_segments),
    )
    return AudioSource(
        source_type="youtube",
        source_id=video_id,
        url=url,
        title=info.get("title", ""),
        audio_path=str(wav_path.resolve()),
        duration_sec=float(info.get("duration", 0.0)),
        chapters=chapters,
        sponsor_segments=sponsor_segments,
        extra={"uploader": info.get("uploader", "")},
    )


def download_podcast_episode(
    audio_url: str,
    *,
    title: str,
    output_dir: Path,
    episode_guid: str | None = None,
) -> AudioSource:
    """Download a single podcast episode (direct mp3/m4a/etc URL) and convert to wav."""
    import httpx

    output_dir.mkdir(parents=True, exist_ok=True)
    source_id = episode_guid or _stable_id(audio_url)
    raw_path = output_dir / f"{source_id}.audio"
    wav_path = output_dir / f"{source_id}.wav"

    logger.info("audio.downloader.podcast.start", url=audio_url, source_id=source_id)
    with httpx.stream("GET", audio_url, follow_redirects=True, timeout=60.0) as resp:
        resp.raise_for_status()
        with raw_path.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                f.write(chunk)

    _convert_to_pcm_wav(raw_path, wav_path)
    raw_path.unlink(missing_ok=True)

    duration_sec = _wav_duration(wav_path)
    logger.info(
        "audio.downloader.podcast.done",
        url=audio_url,
        source_id=source_id,
        duration_sec=duration_sec,
    )
    return AudioSource(
        source_type="podcast",
        source_id=source_id,
        url=audio_url,
        title=title,
        audio_path=str(wav_path.resolve()),
        duration_sec=duration_sec,
    )


def normalize_local_file(path: Path, *, output_dir: Path) -> AudioSource:
    """Adopt a local audio file: convert to 16 kHz mono wav if needed."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_id = _stable_id(str(path.resolve()))
    wav_path = output_dir / f"{source_id}.wav"
    _convert_to_pcm_wav(path, wav_path)
    return AudioSource(
        source_type="local",
        source_id=source_id,
        url=str(path.resolve()),
        title=path.stem,
        audio_path=str(wav_path.resolve()),
        duration_sec=_wav_duration(wav_path),
    )


def parse_podcast_feed(feed_url: str) -> list[tuple[str, str, str]]:
    """Return `(audio_url, title, guid)` triples for every episode in an RSS feed.

    The caller passes each triple to `download_podcast_episode` individually so
    the orchestrator can resume mid-feed and parallelise across episodes.
    """
    try:
        import feedparser
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "feedparser not installed. Install the audio extra: `uv sync --extra audio`."
        ) from exc

    parsed = feedparser.parse(feed_url)
    episodes: list[tuple[str, str, str]] = []
    for entry in parsed.entries:
        enclosure = next(
            (
                link
                for link in entry.get("links", [])
                if link.get("rel") == "enclosure" and link.get("type", "").startswith("audio/")
            ),
            None,
        )
        if not enclosure:
            continue
        guid = entry.get("id") or entry.get("guid") or _stable_id(enclosure["href"])
        episodes.append((enclosure["href"], entry.get("title", ""), guid))
    return episodes


def download(url_or_path: str, *, output_dir: Path) -> AudioSource:
    """Dispatch to the right downloader based on the input string."""
    if _YOUTUBE_RE.search(url_or_path):
        return download_youtube(url_or_path, output_dir=output_dir)
    path = Path(url_or_path)
    if path.exists():
        return normalize_local_file(path, output_dir=output_dir)
    # Assume direct audio URL.
    return download_podcast_episode(
        url_or_path, title=Path(url_or_path).stem, output_dir=output_dir
    )


def _wav_duration(path: Path) -> float:
    """Return wav duration in seconds via soundfile (lazy-imported)."""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "soundfile not installed. Install the audio extra: `uv sync --extra audio`."
        ) from exc
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)

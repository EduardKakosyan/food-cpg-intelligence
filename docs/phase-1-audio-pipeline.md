# Phase 1 — Audio Ingestion Pipeline

**Status:** scaffolded; awaiting first end-to-end run on real audio.
**See also:** `docs/research-voice-rag-pivot.md` (the strategic decision this phase implements).

This phase converts Peter Chapman's podcasts and YouTube videos into two
artifact streams from the same processed transcripts:

1. **SFT JSONL** — voice-fine-tune training data
2. **RAG JSONL** — chunks ready to ingest into the sku-food Supabase corpus

Both files for one episode share the same `source_id`, so any downstream review
or correction stays linked across the two outputs.

## Pipeline stages

```
URL or local file
  -> downloader      yt-dlp / RSS + ffmpeg -> 16 kHz mono wav
  -> asr             WhisperX + faster-whisper (large-v3-turbo)
  -> diarization     pyannote.audio 3.1
  -> speaker_id      ECAPA-TDNN match against Peter reference clip
  -> pipeline.align  pair ASR segments with diarized turns
  -> sponsor skip    drop segments inside SponsorBlock spans
  -> filters         duration / ASR / no-speech / compression / filler /
                     overlap / speaker-conf / topic gates
  -> SFT + RAG JSONL  one file each per episode under data/processed/audio/
```

## Install

The heavy ML dependencies live in an optional `audio` extra so the default
install stays light.

```bash
uv sync --extra audio
brew install ffmpeg   # required by the downloader
```

## One-time setup

### 1. Hugging Face token for pyannote

pyannote 3.1 is gated. Accept the model license at
<https://hf.co/pyannote/speaker-diarization-3.1>, then export your token:

```bash
echo "FCPG_HUGGINGFACE_TOKEN=hf_xxx" >> .env
```

### 2. Peter reference clip

Speaker identification needs a 30–60 second clip of clean Peter-only audio
(no host, no music) as 16 kHz mono wav. Any solo monologue from a SKUFood
podcast intro works:

```bash
ffmpeg -i some-episode.mp3 -ss 00:00:15 -t 45 \
       -ac 1 -ar 16000 -vn -f wav \
       data/raw/audio/peter_reference.wav

uv run fcpg audio verify-reference
```

`verify-reference` exits 0 only if the file is 16 kHz mono and ≥10s long.

### 3. (Optional) Override settings

All audio settings live in `configs/audio.yaml`. Filter thresholds,
ASR model, diarization device, speaker-id threshold, and topic seed prompts
are all tunable there. Defaults come straight from `docs/research-voice-rag-pivot.md` §5.

## Run a single source

```bash
uv run fcpg audio process-url "https://www.youtube.com/watch?v=EXAMPLE"
# or
uv run fcpg audio process-url /path/to/local.mp3
# or
uv run fcpg audio process-url "https://example.com/podcast-feed/episode-42.mp3"
```

Output (per episode) in `data/processed/audio/`:

| File | Contents |
|---|---|
| `<source_id>.sft.jsonl` | One `SFTExample` per line — `(system, user, assistant)` chat-format training data |
| `<source_id>.rag.jsonl` | One `RAGChunk` per line — content + timestamp + source url, for sku-food ingestion |
| `<source_id>.episode.json` | Debug summary: speaker verdict, counts, filter-rejection breakdown |

## Run a batch

```bash
# urls.txt: one URL or local path per line
uv run fcpg audio process-batch urls.txt
```

The batch processor loads the Whisper / pyannote / ECAPA models once and reuses
them across episodes, which is the only way ASR latency stays sane on long runs.

## Expected timings (one ~60-minute episode)

| Stage | M4 Pro 48GB | A100 80GB |
|---|---|---|
| Download + ffmpeg | 30s – 2m (network) | same |
| WhisperX `large-v3-turbo` | 5 – 10m | 30 – 60s |
| pyannote diarization | 10 – 15m (CPU only) | 1 – 2m |
| Speaker-ID centroids | ~30s | <10s |
| Align + filter + write | <10s | same |

→ M4 Pro end-to-end: ~20–30 min per episode. A100: 2–4 min.

For 100h of audio plan for one A100 day at most, well under $10 on Lambda.

## Schema reference

`food_cpg_intelligence.ingest.audio.schemas`:

- `AudioSource` — provenance + path to normalised wav
- `Word`, `TranscribedSegment` — ASR output with word-level alignment scores
- `DiarizedTurn` — pyannote turn with `overlap_ratio` against other speakers
- `AlignedSegment` — ASR segment after diarization assignment
- `PeterSegment` — speaker-identified Peter-only segment
- `FilterReport` — per-filter pass/fail, with `passed` and `rejection_reasons()`
- `SFTExample`, `RAGChunk` — the two output formats

All models are immutable Pydantic (`frozen=True`) and `model_dump_json()` is
JSONL-safe (no embedded newlines).

## Filter tuning

`FilterThresholds` defaults (in `filters.py`, mirrored in `configs/audio.yaml`):

| Filter | Default | Rationale |
|---|---|---|
| `min_duration_sec` | 3.0 | Short turns under-train style and over-fit single words |
| `max_duration_sec` | 90.0 | Long turns over-pack context; split before training if useful |
| `min_asr_confidence` | 0.6 | Mean word-level alignment score |
| `max_no_speech_prob` | 0.3 | Whisper's own silence-likelihood |
| `max_compression_ratio` | 2.4 | Token-loop hallucination signature |
| `max_filler_ratio` | 0.15 | Drop banter / filler-heavy turns |
| `max_overlap_ratio` | 0.10 | Crosstalk corrupts training signal |
| `min_speaker_confidence` | 0.55 | Cosine to the Peter reference embedding |
| `relative_topic_percentile` | 0.40 | Keep the top 60% of corpus by Food-&-CPG seed-prompt similarity |

Tune in `configs/audio.yaml`. The relative topic gate is applied per-episode
in this version; a global re-rank across the whole corpus is a later step
(Phase 2) once we have all transcripts.

## Outputs that still need wiring up

These are documented gaps in the v1 scaffold — fill them in as we get further:

- **Topic-similarity scorer** (`_maybe_topic_score` in `pipeline.py`) returns
  zeros today. Wire in a real `BAAI/bge-large-en-v1.5` cosine against the
  seed prompts before relying on the topic gate.
- **Style-rewrite, reverse-QA, RAG-grounded SFT modes** — only `continuation`
  mode is generated. The 60/30/10 mix from the research synthesis comes
  online when we add the Claude-driven synthesis stage (Phase 2).
- **Entity scrubbing** — `SFTExample.scrubbed = False` everywhere. Wire in
  the `[number]`, `[year]`, `[retailer]` placeholder pass before Lambda
  training (the largest data-side lever per Stream 2).
- **Cache layer** — re-running on the same `source_id` re-runs all stages.
  Add an artifact-hash cache once we're past prototype.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: No module named 'whisperx'` | `uv sync --extra audio` |
| `ffmpeg binary not found on PATH` | `brew install ffmpeg` (macOS) |
| pyannote raises `Unauthorized` | Accept the license on hf.co and set `FCPG_HUGGINGFACE_TOKEN` |
| `Expected 16 kHz mono wav, got 44100 Hz` in speaker-id | Re-run `download` — your wav wasn't normalised; check ffmpeg output |
| Verdict has no Peter cluster (all `cluster_similarities` < 0.55) | Either Peter isn't in this audio or your reference clip is bad — re-record a longer, cleaner reference |
| Whisper produces "thank you for watching" / "subscribe" loops | Hallucination on silent regions; the `compression_ratio` + repeat-regex filters should drop these, raise `max_compression_ratio` lower if they leak through |
| Diarization runs 30+ minutes on a 30-minute episode | Expected on CPU. Run on A100 if iterating |

## How this feeds the next phases

- **Phase 2 (Claude synthesis):** consumes `<source_id>.sft.jsonl` to generate
  style-rewrite pairs and reverse-Q&A, expanding the 100%-continuation data
  into the 60/30/10 mix.
- **Phase 3 (LoRA fine-tune):** consumes the merged JSONL on Lambda.
- **Phase 5 (sku-food ingestion):** consumes `<source_id>.rag.jsonl`, fed
  through the sku-food `pnpm re-ingest:local` pipeline so transcripts join
  newsletters in the same Supabase corpus.

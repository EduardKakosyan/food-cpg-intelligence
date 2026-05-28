"""Continuation-mode SFT example generator.

For voice transfer, the highest-signal training data is "Peter prose as the
assistant turn, generic eliciting prompt as the user turn." The LoRA learns
the rhythm, framing, and word choice — not specific facts.

Pipeline (per ProcessedDocument):
    1. Scrub entities (numbers, retailers, brands, dates) -> placeholders
    2. Chunk the scrubbed text into 512-2048-token paragraph-aligned segments
    3. Pair each chunk with a randomly-selected eliciting prompt, using the
       document title (or first heading) as the {topic} slot
    4. Emit a TrainingTriple per chunk with generation_mode="continuation"
       and scrubbed=True

Token counting is approximate: we use words * 1.33 as a proxy for tokens, then
clamp via paragraph boundaries. The actual Qwen tokenizer is loaded only when
the consumer wants exact token counts at write time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import structlog

from food_cpg_intelligence.data.models import ProcessedDocument
from food_cpg_intelligence.training.models import TrainingTriple
from food_cpg_intelligence.training.scrubbing import scrub_text

logger = structlog.stdlib.get_logger(__name__)

# Approximate tokens per word for Qwen 3.5 tokenizer (empirically ~1.33).
_TOKENS_PER_WORD = 1.33


@dataclass(frozen=True)
class ChunkConfig:
    """Tunable parameters for chunking + eliciting prompt selection."""

    target_tokens: int = 1200
    min_tokens: int = 256  # drop tiny tail chunks; voice signal too weak
    max_tokens: int = 2048
    seed: int = 42


# Eliciting-prompt templates. Each must contain a `{topic}` slot so the user
# message stays anchored to the doc topic but the model can't lean on the
# prompt to derive content. Variations prevent the LoRA from over-fitting to
# one phrasing.
_ELICITING_TEMPLATES: tuple[str, ...] = (
    "Walk me through how you think about {topic}.",
    "What's your take on {topic}?",
    "Can you share your perspective on {topic}?",
    "How should I think about {topic}?",
    "From your experience, what's important about {topic}?",
    "Tell me how you'd approach {topic}.",
    "What do food and beverage SMEs need to know about {topic}?",
    "How does {topic} actually play out in retail?",
)


def _approx_tokens(text: str) -> int:
    """Word count * approximate tokens-per-word for fast chunking decisions."""
    return int(len(text.split()) * _TOKENS_PER_WORD)


def _split_paragraphs(text: str) -> list[str]:
    """Split text on double-newline; collapse empty paragraphs."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _topic_from_document(doc: ProcessedDocument) -> str:
    """Derive a short eliciting-prompt topic from the document.

    Prefers the title; falls back to the first non-empty line. Strips trailing
    punctuation and clamps length so the prompt reads naturally.
    """
    candidate = (doc.title or "").strip()
    if not candidate:
        first_line = next(
            (line.strip() for line in doc.full_text.splitlines() if line.strip()),
            "",
        )
        candidate = first_line[:80]

    # Remove trailing punctuation so "{topic}." doesn't become "topic.?"
    candidate = candidate.rstrip("?!.,:;\"' ")

    if not candidate:
        return "this topic"

    # Lowercase the first letter so "{topic}" reads cleanly mid-sentence
    return candidate[0].lower() + candidate[1:] if candidate else candidate


def _greedy_chunk(paragraphs: list[str], config: ChunkConfig) -> list[str]:
    """Greedy paragraph-aligned chunker.

    Walks paragraphs in order; appends to the current chunk while adding it
    keeps the chunk under `target_tokens`. When a paragraph would push over,
    closes the current chunk and starts a new one. Over-long single paragraphs
    are emitted as-is (we don't sentence-split mid-paragraph — paragraph breaks
    are content boundaries Peter usually places intentionally).

    Returns chunks whose approximate token count is in [min_tokens, max_tokens].
    Chunks shorter than min_tokens are dropped to keep voice signal strong.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = _approx_tokens(para)
        if not current:
            current.append(para)
            current_tokens = para_tokens
            continue

        # Would adding this paragraph exceed the target?
        if current_tokens + para_tokens > config.target_tokens:
            chunks.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
        else:
            current.append(para)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))

    # Drop chunks below the minimum; cap any that ended up over max.
    out: list[str] = []
    for c in chunks:
        tokens = _approx_tokens(c)
        if tokens < config.min_tokens:
            continue
        out.append(c)
    return out


def generate_from_document(
    doc: ProcessedDocument,
    *,
    config: ChunkConfig | None = None,
    rng: random.Random | None = None,
) -> list[TrainingTriple]:
    """Generate one or more continuation TrainingTriples from a single document.

    Returns an empty list when the document has no eligible chunks (too short
    after scrubbing, or all chunks below `min_tokens`).
    """
    cfg = config or ChunkConfig()
    rand = rng or random.Random(cfg.seed)

    if not doc.full_text.strip():
        return []

    scrubbed = scrub_text(doc.full_text)
    paragraphs = _split_paragraphs(scrubbed.text)
    if not paragraphs:
        return []

    chunks = _greedy_chunk(paragraphs, cfg)
    if not chunks:
        return []

    topic = _topic_from_document(doc)
    triples: list[TrainingTriple] = []
    for idx, chunk in enumerate(chunks):
        template = rand.choice(_ELICITING_TEMPLATES)
        instruction = template.format(topic=topic)
        triples.append(
            TrainingTriple(
                triple_id=f"continuation-{doc.doc_id}-{idx:03d}",
                instruction=instruction,
                context="",
                response=chunk,
                source_doc_id=doc.doc_id,
                category=doc.content_type,
                question_type="advice",
            )
        )

    return triples


def generate_from_corpus(
    docs: list[ProcessedDocument],
    *,
    config: ChunkConfig | None = None,
) -> list[TrainingTriple]:
    """Generate continuation TrainingTriples from a whole corpus.

    Stable RNG: each document seeds its own per-doc prompt selection from a
    deterministic offset so two runs over the same corpus produce identical
    output.
    """
    cfg = config or ChunkConfig()
    triples: list[TrainingTriple] = []
    for i, doc in enumerate(docs):
        rng = random.Random(cfg.seed + i)
        triples.extend(generate_from_document(doc, config=cfg, rng=rng))
    logger.info(
        "continuation_generated",
        documents=len(docs),
        triples=len(triples),
        avg_per_doc=round(len(triples) / max(1, len(docs)), 2),
    )
    return triples

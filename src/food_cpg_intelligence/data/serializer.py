"""Serialize and deserialize processed documents (JSONL, Parquet, manifest)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import structlog

from food_cpg_intelligence.data.models import ProcessedDocument

logger = structlog.stdlib.get_logger(__name__)


def _flatten_for_parquet(doc: ProcessedDocument) -> dict[str, object]:
    """Build a flat dict from a ProcessedDocument, expanding metadata into meta_* columns."""
    rec: dict[str, object] = {k: v for k, v in doc.model_dump().items() if k != "metadata"}
    for k, v in doc.metadata.items():
        rec[f"meta_{k}"] = v
    return rec


def save_jsonl(docs: list[ProcessedDocument], path: Path) -> Path:
    """Write a list of ProcessedDocuments as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.model_dump_json() + "\n")
    logger.info("saved_jsonl", path=str(path), count=len(docs))
    return path


def load_jsonl(path: Path) -> list[ProcessedDocument]:
    """Load ProcessedDocuments from a JSONL file."""
    docs: list[ProcessedDocument] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                docs.append(ProcessedDocument.model_validate_json(stripped))
    return docs


def save_parquet(docs: list[ProcessedDocument], path: Path) -> Path:
    """Save documents as a Parquet file via Polars for columnar analytics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [_flatten_for_parquet(doc) for doc in docs]
    df = pl.DataFrame(records)
    df.write_parquet(path)
    logger.info("saved_parquet", path=str(path), rows=len(df))
    return path


def save_manifest(docs: list[ProcessedDocument], path: Path) -> Path:
    """Generate and save a corpus_manifest.json with corpus statistics."""
    path.parent.mkdir(parents=True, exist_ok=True)

    total_words = sum(d.word_count for d in docs)
    by_type: dict[str, int] = {}
    for d in docs:
        by_type[d.content_type] = by_type.get(d.content_type, 0) + 1

    manifest = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "total_documents": len(docs),
        "total_words": total_words,
        "avg_words_per_doc": round(total_words / len(docs), 1) if docs else 0,
        "by_content_type": by_type,
        "documents": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "word_count": d.word_count,
                "source_file": d.source_file,
            }
            for d in docs
        ],
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("saved_manifest", path=str(path), total_docs=len(docs))
    return path


def save_processed_documents(docs: list[ProcessedDocument], output_dir: Path) -> Path:
    """Save processed documents in all formats (JSONL + Parquet + manifest)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(docs, output_dir / "corpus.jsonl")
    save_parquet(docs, output_dir / "corpus.parquet")
    save_manifest(docs, output_dir / "corpus_manifest.json")
    return output_dir

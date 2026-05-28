"""Orchestrates the MVP voice-fine-tune dataset build.

Pipeline:
    corpus.jsonl
      -> continuation.generate_from_corpus (chunk + scrub + eliciting prompt)
      -> quality_filter.run_quality_pipeline (length / dedup / contamination / balance)
      -> formatter.save_formatted_dataset (90/10 split, Qwen ChatML)
      -> data/training/sft_mvp/{train,val}.jsonl + dataset_meta.json
                                + inference_test_faq.jsonl  (post-train smoke set)
                                + build_report.json         (audit trail)

Designed to run end-to-end from one CLI command: `fcpg train build-mvp-dataset`.
Deterministic via the seeds in ChunkConfig and the formatter's RNG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from food_cpg_intelligence.data.serializer import load_jsonl
from food_cpg_intelligence.evaluation.models import GoldStandardSet
from food_cpg_intelligence.training.continuation import ChunkConfig, generate_from_corpus
from food_cpg_intelligence.training.formatter import save_formatted_dataset
from food_cpg_intelligence.training.models import TrainingTriple, compute_stats
from food_cpg_intelligence.training.quality_filter import FilterThresholds, run_quality_pipeline

# Defaults sized for continuation-mode chunks. ChunkConfig.target_tokens is in
# tokens; chars approximately 4.2x tokens, so a 1200-token chunk is ~5000 chars.
# We allow up to 10000 chars to cover the top of the 2048-token range with margin.
_CONTINUATION_THRESHOLDS = FilterThresholds(
    min_response_chars=400,
    max_response_chars=10000,
    min_instruction_chars=10,
    max_category_fraction=0.95,
)

logger = structlog.stdlib.get_logger(__name__)


@dataclass(frozen=True)
class BuildConfig:
    """Inputs and outputs for the MVP build."""

    corpus_path: Path
    gold_standard_path: Path
    faq_pairs_path: Path
    output_dir: Path
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    train_fraction: float = 0.9
    split_seed: int = 42


def _load_gold_standard(path: Path) -> GoldStandardSet:
    """Load the gold standard test set for contamination filtering."""
    if not path.exists():
        logger.warning("gold_standard_missing", path=str(path))
        return GoldStandardSet()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return GoldStandardSet.model_validate(raw)


def _stage_faq_inference_set(faq_path: Path, output_path: Path) -> int:
    """Copy FAQ questions to the output dir as an inference smoke-test set.

    The FAQ pairs ship with empty responses (they were question sources, not
    Peter-validated Q&A pairs). We pass through `instruction` + `context` only
    so a post-train sampler can feed each question and let a human or judge
    score the model's Peter-voice response.
    """
    if not faq_path.exists():
        logger.warning("faq_pairs_missing", path=str(faq_path))
        return 0

    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        faq_path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dst.write(
                json.dumps(
                    {
                        "test_id": obj.get("triple_id", ""),
                        "instruction": obj.get("instruction", ""),
                        "context": obj.get("context", ""),
                        "category": obj.get("category", ""),
                        "peter_framing": obj.get("peter_framing", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    logger.info("faq_inference_set_written", count=written, path=str(output_path))
    return written


def _write_report(
    config: BuildConfig,
    raw_triples: list[TrainingTriple],
    filtered_triples: list[TrainingTriple],
    filter_stats: dict[str, int],
    train_path: Path,
    val_path: Path,
    faq_count: int,
) -> Path:
    """Audit trail: parameters + input/output counts + per-category stats."""
    report_path = config.output_dir / "build_report.json"
    raw_stats = compute_stats(raw_triples)
    final_stats = compute_stats(
        filtered_triples,
        dedup_removed=filter_stats.get("dedup_removed", 0),
        contamination_removed=filter_stats.get("contamination_removed", 0),
    )
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "config": {
            "corpus_path": str(config.corpus_path),
            "gold_standard_path": str(config.gold_standard_path),
            "faq_pairs_path": str(config.faq_pairs_path),
            "output_dir": str(config.output_dir),
            "chunk": {
                "target_tokens": config.chunk.target_tokens,
                "min_tokens": config.chunk.min_tokens,
                "max_tokens": config.chunk.max_tokens,
                "seed": config.chunk.seed,
            },
            "train_fraction": config.train_fraction,
            "split_seed": config.split_seed,
        },
        "raw": raw_stats.model_dump(),
        "final": final_stats.model_dump(),
        "filter_pipeline": filter_stats,
        "outputs": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "inference_test_faq_path": str(config.output_dir / "inference_test_faq.jsonl"),
            "faq_count": faq_count,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("build_report_written", path=str(report_path))
    return report_path


def build_mvp_dataset(config: BuildConfig) -> dict[str, Path]:
    """Run the full MVP dataset build end-to-end.

    Returns a dict of output paths so callers (the CLI) can echo them.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "mvp_build_start",
        **{
            "corpus": str(config.corpus_path),
            "output": str(config.output_dir),
        },
    )

    docs = load_jsonl(config.corpus_path)
    if not docs:
        raise ValueError(f"No documents loaded from {config.corpus_path}")

    raw_triples = generate_from_corpus(docs, config=config.chunk)
    if not raw_triples:
        raise ValueError("No training triples produced from corpus")

    gold = _load_gold_standard(config.gold_standard_path)
    filtered, filter_stats = run_quality_pipeline(
        raw_triples, list(gold.items), thresholds=_CONTINUATION_THRESHOLDS
    )

    if not filtered:
        raise ValueError("Quality pipeline removed every triple — review filters")

    train_path, val_path = save_formatted_dataset(
        filtered,
        config.output_dir,
        seed=config.split_seed,
        train_fraction=config.train_fraction,
    )

    faq_path = config.output_dir / "inference_test_faq.jsonl"
    faq_count = _stage_faq_inference_set(config.faq_pairs_path, faq_path)

    report_path = _write_report(
        config,
        raw_triples,
        filtered,
        filter_stats,
        train_path,
        val_path,
        faq_count,
    )

    logger.info(
        "mvp_build_complete",
        raw=len(raw_triples),
        kept=len(filtered),
        train=int(len(filtered) * config.train_fraction),
        val=len(filtered) - int(len(filtered) * config.train_fraction),
        faq_smoke=faq_count,
    )
    return {
        "train": train_path,
        "val": val_path,
        "inference_test_faq": faq_path,
        "report": report_path,
    }

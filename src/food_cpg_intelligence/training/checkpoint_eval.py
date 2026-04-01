"""Evaluate training checkpoints against the gold standard set.

Uses mlx-lm generate for inference, or Ollama if the model is registered.
Provides quick directional metrics without full LLM-as-judge evaluation.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import structlog

from food_cpg_intelligence.evaluation.gold_set_generator import load_gold_items
from food_cpg_intelligence.evaluation.models import GoldStandardItem

logger = structlog.stdlib.get_logger(__name__)


def _keyword_overlap(reference: str, response: str) -> float:
    """Compute keyword overlap ratio between reference and response."""
    ref_words = {w.lower() for w in reference.split() if len(w) > 3}
    resp_words = {w.lower() for w in response.split() if len(w) > 3}
    if not ref_words:
        return 0.0
    return len(ref_words & resp_words) / len(ref_words)


def _sample_questions(
    gold_items: list[GoldStandardItem],
    sample_size: int,
    *,
    seed: int = 42,
) -> list[GoldStandardItem]:
    """Sample questions from the gold set, stratified by category."""
    rng = random.Random(seed)
    sampled = rng.sample(gold_items, min(sample_size, len(gold_items)))
    return sampled


def evaluate_checkpoint_mlx(
    base_model: str,
    adapter_path: Path,
    gold_items: list[GoldStandardItem],
    *,
    sample_size: int = 50,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate a checkpoint using mlx-lm generate.

    Args:
        base_model: HuggingFace model ID for the base model.
        adapter_path: Path to the LoRA adapter checkpoint.
        gold_items: Gold standard items to evaluate against.
        sample_size: Number of questions to sample.
        seed: Random seed for sampling.

    Returns:
        Dict with metrics: avg_keyword_overlap, avg_response_length, avg_latency_ms, sample_count.
    """
    from mlx_lm import generate, load

    samples = _sample_questions(gold_items, sample_size, seed=seed)

    logger.info(
        "checkpoint_eval_start",
        adapter=str(adapter_path),
        samples=len(samples),
    )

    model, tokenizer = load(base_model, adapter_path=str(adapter_path))  # type: ignore[misc]

    overlaps: list[float] = []
    lengths: list[int] = []
    latencies: list[float] = []

    for item in samples:
        prompt = f"<|im_start|>user\n{item.question}<|im_end|>\n<|im_start|>assistant\n"

        start = time.perf_counter()
        response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
        elapsed_ms = (time.perf_counter() - start) * 1000

        overlaps.append(_keyword_overlap(item.reference_answer, response))
        lengths.append(len(response.split()))
        latencies.append(elapsed_ms)

    return {
        "adapter_path": str(adapter_path),
        "sample_count": len(samples),
        "avg_keyword_overlap": round(sum(overlaps) / len(overlaps), 3) if overlaps else 0,
        "avg_response_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
    }


def evaluate_checkpoint_ollama(
    model_name: str,
    gold_items: list[GoldStandardItem],
    *,
    sample_size: int = 50,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluate a registered Ollama model against gold standard questions."""
    import httpx

    samples = _sample_questions(gold_items, sample_size, seed=seed)

    logger.info("checkpoint_eval_ollama", model=model_name, samples=len(samples))

    overlaps: list[float] = []
    lengths: list[int] = []
    latencies: list[float] = []

    client = httpx.Client(timeout=120.0)

    for item in samples:
        start = time.perf_counter()
        resp = client.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": item.question, "stream": False},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            response_text = resp.json().get("response", "")
            overlaps.append(_keyword_overlap(item.reference_answer, response_text))
            lengths.append(len(response_text.split()))
            latencies.append(elapsed_ms)
        else:
            logger.warning("ollama_error", status=resp.status_code, question=item.question_id)

    client.close()

    return {
        "model": model_name,
        "sample_count": len(samples),
        "avg_keyword_overlap": round(sum(overlaps) / len(overlaps), 3) if overlaps else 0,
        "avg_response_length": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
    }


def evaluate_all_checkpoints(
    checkpoints_dir: Path,
    base_model: str,
    gold_set_path: Path,
    *,
    sample_size: int = 50,
) -> list[dict[str, object]]:
    """Evaluate all adapter checkpoints in a directory.

    Expects checkpoint subdirectories like: checkpoints_dir/checkpoint-100/, checkpoint-200/, etc.
    """
    gold_items = load_gold_items(gold_set_path)
    if not gold_items:
        raise ValueError(f"No gold items found at {gold_set_path}")

    # Find checkpoint dirs (sorted by step number)
    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {checkpoints_dir}")

    results: list[dict[str, object]] = []
    for ckpt_dir in checkpoint_dirs:
        logger.info("evaluating_checkpoint", checkpoint=ckpt_dir.name)
        try:
            result = evaluate_checkpoint_mlx(
                base_model, ckpt_dir, gold_items, sample_size=sample_size
            )
            results.append(result)
        except Exception as exc:
            logger.error("checkpoint_eval_failed", checkpoint=ckpt_dir.name, error=str(exc))

    # Sort by keyword overlap (best first)
    results.sort(key=lambda r: float(str(r.get("avg_keyword_overlap", 0))), reverse=True)
    return results


def format_eval_table(results: list[dict[str, object]]) -> str:
    """Format evaluation results as a readable table."""
    if not results:
        return "No results to display."

    lines = [
        "| Checkpoint | Keyword Overlap | Avg Length | Avg Latency (ms) |",
        "|------------|----------------|------------|------------------|",
    ]
    for r in results:
        adapter = Path(str(r.get("adapter_path", r.get("model", "?")))).name
        lines.append(
            f"| {adapter:10s} | {r.get('avg_keyword_overlap', 0):14.3f} "
            f"| {r.get('avg_response_length', 0):>10} "
            f"| {r.get('avg_latency_ms', 0):>16.0f} |"
        )
    return "\n".join(lines)

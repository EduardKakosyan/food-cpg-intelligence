"""Evaluation pipeline runner — generate responses, judge, and report.

Orchestrates the full benchmark: fine-tuned model (Ollama) vs Claude baseline,
with blind LLM-as-judge scoring and head-to-head comparison reporting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from food_cpg_intelligence.evaluation.blind_eval import (
    prepare_blind_batch,
    reveal_results,
)
from food_cpg_intelligence.evaluation.gold_set_generator import load_gold_set
from food_cpg_intelligence.evaluation.judge import (
    build_judge_prompt,
    parse_judge_output,
    save_judge_scores,
    scores_dict_to_judge_score,
)
from food_cpg_intelligence.evaluation.models import (
    BlindBatch,
    EvalResponse,
    GoldStandardItem,
    JudgeScore,
)
from food_cpg_intelligence.evaluation.report import (
    compile_report,
    save_report,
)
from food_cpg_intelligence.training.formatter import SYSTEM_PROMPT

logger = structlog.stdlib.get_logger(__name__)


# ---------------------------------------------------------------------------
# I/O utilities
# ---------------------------------------------------------------------------


def save_eval_responses(responses: list[EvalResponse], path: Path) -> Path:
    """Save EvalResponse objects to JSONL (overwrites)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for resp in responses:
            f.write(resp.model_dump_json() + "\n")
    return path


def _append_eval_response(resp: EvalResponse, path: Path) -> None:
    """Append a single EvalResponse to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(resp.model_dump_json() + "\n")


def load_eval_responses(path: Path) -> list[EvalResponse]:
    """Load EvalResponse objects from JSONL. Skips malformed lines."""
    responses: list[EvalResponse] = []
    if not path.exists():
        return responses
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                responses.append(EvalResponse.model_validate_json(stripped))
            except Exception as exc:
                logger.warning("response_parse_error", line=line_num, error=str(exc))
    return responses


def get_completed_ids(path: Path) -> set[str]:
    """Return question_ids already present in a responses/scores JSONL file."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
                qid = data.get("question_id") or data.get("blind_id", "")
                if qid:
                    ids.add(qid)
            except Exception:
                pass
    return ids


# ---------------------------------------------------------------------------
# Response generation — Ollama (fine-tuned model)
# ---------------------------------------------------------------------------


def generate_responses_ollama(
    gold_items: list[GoldStandardItem],
    *,
    model_name: str = "skufood-9b-v2",
    system_prompt: str = SYSTEM_PROMPT,
    output_path: Path,
    timeout: float = 120.0,
) -> list[EvalResponse]:
    """Generate responses from Ollama for all gold standard items.

    Resumes from where it left off if output_path already has responses.
    """
    completed = get_completed_ids(output_path)
    remaining = [item for item in gold_items if item.question_id not in completed]
    total = len(gold_items)

    if not remaining:
        logger.info("ollama_all_complete", total=total)
        return load_eval_responses(output_path)

    logger.info(
        "ollama_start",
        model=model_name,
        total=total,
        remaining=len(remaining),
        skipped=len(completed),
    )

    client = httpx.Client(timeout=timeout)

    for i, item in enumerate(remaining):
        start = time.perf_counter()
        try:
            resp = client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item.question},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            response_text = resp.json()["message"]["content"]
        except Exception as exc:
            logger.error(
                "ollama_error",
                question_id=item.question_id,
                error=str(exc),
            )
            continue

        elapsed_ms = (time.perf_counter() - start) * 1000

        eval_resp = EvalResponse(
            question_id=item.question_id,
            system="finetune",
            response=response_text,
            latency_ms=round(elapsed_ms, 1),
        )
        _append_eval_response(eval_resp, output_path)

        done = len(completed) + i + 1
        if done % 10 == 0 or done == total:
            logger.info("ollama_progress", completed=done, total=total)

    client.close()
    return load_eval_responses(output_path)


# ---------------------------------------------------------------------------
# Response generation — Claude (baseline)
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _call_claude(
    client: anthropic.Anthropic,
    *,
    model: str,
    system_prompt: str,
    question: str,
) -> tuple[str, float]:
    """Call Claude API with retry. Returns (response_text, latency_ms)."""
    start = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    return message.content[0].text, elapsed_ms


def generate_responses_claude(
    gold_items: list[GoldStandardItem],
    *,
    model: str = "claude-sonnet-4-6",
    system_prompt: str = SYSTEM_PROMPT,
    output_path: Path,
    api_key: str,
) -> list[EvalResponse]:
    """Generate responses from Claude API for all gold standard items.

    Resumes from where it left off if output_path already has responses.
    """
    completed = get_completed_ids(output_path)
    remaining = [item for item in gold_items if item.question_id not in completed]
    total = len(gold_items)

    if not remaining:
        logger.info("claude_all_complete", total=total)
        return load_eval_responses(output_path)

    logger.info(
        "claude_start",
        model=model,
        total=total,
        remaining=len(remaining),
        skipped=len(completed),
    )

    client = anthropic.Anthropic(api_key=api_key)

    for i, item in enumerate(remaining):
        try:
            response_text, elapsed_ms = _call_claude(
                client,
                model=model,
                system_prompt=system_prompt,
                question=item.question,
            )
        except Exception as exc:
            logger.error(
                "claude_error",
                question_id=item.question_id,
                error=str(exc),
            )
            continue

        eval_resp = EvalResponse(
            question_id=item.question_id,
            system="baseline",
            response=response_text,
            latency_ms=round(elapsed_ms, 1),
        )
        _append_eval_response(eval_resp, output_path)

        done = len(completed) + i + 1
        if done % 10 == 0 or done == total:
            logger.info("claude_progress", completed=done, total=total)

    return load_eval_responses(output_path)


# ---------------------------------------------------------------------------
# Blind LLM-as-judge scoring (Message Batches API — 50% cost reduction)
# ---------------------------------------------------------------------------


def run_judge_scoring(
    blind_batch: BlindBatch,
    *,
    judge_model: str = "claude-sonnet-4-6",
    api_key: str,
    output_path: Path,
    poll_interval: float = 30.0,
) -> list[JudgeScore]:
    """Score all blind items using Claude Message Batches API.

    Uses the async batch endpoint for 50% cost savings and no rate limits.
    Resumes from where it left off if output_path already has scores.
    """
    completed = get_completed_ids(output_path)
    remaining = [item for item in blind_batch.items if item.blind_id not in completed]
    total = len(blind_batch.items)

    if not remaining:
        logger.info("judge_all_complete", total=total)
        return _load_judge_scores_raw(output_path)

    logger.info(
        "judge_batch_start",
        judge_model=judge_model,
        total=total,
        remaining=len(remaining),
        skipped=len(completed),
    )

    client = anthropic.Anthropic(api_key=api_key)

    # Build batch requests
    batch_requests = []
    for item in remaining:
        prompt = build_judge_prompt(
            question=item.question,
            reference_answer=item.reference_answer,
            response=item.response,
        )
        batch_requests.append(
            anthropic.types.messages.batch_create_params.Request(
                custom_id=item.blind_id,
                params=anthropic.types.messages.MessageCreateParamsNonStreaming(
                    model=judge_model,
                    max_tokens=512,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                ),
            )
        )

    # Submit batch
    batch = client.messages.batches.create(requests=batch_requests)
    logger.info(
        "judge_batch_submitted",
        batch_id=batch.id,
        requests=len(batch_requests),
    )

    # Poll until complete
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        done = counts.succeeded + counts.errored + counts.expired + counts.canceled
        logger.info(
            "judge_batch_poll",
            batch_id=batch.id,
            status=batch.processing_status,
            succeeded=counts.succeeded,
            errored=counts.errored,
            total=done,
            of=len(batch_requests),
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval)

    # Retrieve results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_count = 0
    error_count = 0

    for result in client.messages.batches.results(batch.id):
        blind_id = result.custom_id

        if result.result.type == "succeeded":
            raw_output = result.result.message.content[0].text
            try:
                scores = parse_judge_output(raw_output)
                judge_score = scores_dict_to_judge_score(
                    scores,
                    question_id=blind_id,
                    system="blind",
                )
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(judge_score.model_dump_json() + "\n")
                parsed_count += 1
            except Exception as exc:
                logger.error("judge_parse_error", blind_id=blind_id, error=str(exc))
                error_count += 1
        else:
            logger.error(
                "judge_batch_item_failed",
                blind_id=blind_id,
                result_type=result.result.type,
            )
            error_count += 1

    logger.info(
        "judge_batch_complete",
        batch_id=batch.id,
        parsed=parsed_count,
        errors=error_count,
    )

    return _load_judge_scores_raw(output_path)


def _load_judge_scores_raw(path: Path) -> list[JudgeScore]:
    """Load JudgeScore objects from JSONL."""
    scores: list[JudgeScore] = []
    if not path.exists():
        return scores
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                scores.append(JudgeScore.model_validate_json(stripped))
            except Exception:
                continue
    return scores


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


def run_full_pipeline(
    gold_set_path: Path,
    output_dir: Path,
    *,
    ollama_model: str = "skufood-9b-v2",
    claude_model: str = "claude-sonnet-4-6",
    judge_model: str = "claude-sonnet-4-6",
    api_key: str,
) -> Path:
    """Run the full 4-step evaluation pipeline.

    Steps:
        1. Generate responses from Ollama (fine-tuned model)
        2. Generate responses from Claude (baseline)
        3. Blind judge all responses
        4. Unmask, compile report, save

    Returns:
        Path to the output directory with all results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gold standard
    gold_set = load_gold_set(gold_set_path)
    gold_items = list(gold_set.items)
    logger.info("pipeline_start", gold_items=len(gold_items), output_dir=str(output_dir))

    # Step 1: Fine-tuned model responses
    logger.info("pipeline_step", step=1, description="Generate finetune responses (Ollama)")
    finetune_path = output_dir / "responses_finetune.jsonl"
    finetune_responses = generate_responses_ollama(
        gold_items,
        model_name=ollama_model,
        output_path=finetune_path,
    )
    logger.info("pipeline_step_done", step=1, responses=len(finetune_responses))

    # Step 2: Claude baseline responses
    logger.info("pipeline_step", step=2, description="Generate baseline responses (Claude)")
    baseline_path = output_dir / "responses_baseline.jsonl"
    baseline_responses = generate_responses_claude(
        gold_items,
        model=claude_model,
        output_path=baseline_path,
        api_key=api_key,
    )
    logger.info("pipeline_step_done", step=2, responses=len(baseline_responses))

    # Step 3: Blind judge
    logger.info("pipeline_step", step=3, description="Blind LLM-as-judge scoring")

    responses_by_system = {
        "finetune": finetune_responses,
        "baseline": baseline_responses,
    }
    blind_batch = prepare_blind_batch(gold_items, responses_by_system)

    # Save blind batch for reproducibility
    batch_path = output_dir / "blind_batch.json"
    batch_path.write_text(blind_batch.model_dump_json(indent=2), encoding="utf-8")

    scores_blind_path = output_dir / "scores_blind.jsonl"
    blind_scores = run_judge_scoring(
        blind_batch,
        judge_model=judge_model,
        api_key=api_key,
        output_path=scores_blind_path,
    )
    logger.info("pipeline_step_done", step=3, scores=len(blind_scores))

    # Step 4: Unmask and report
    logger.info("pipeline_step", step=4, description="Unmask results and compile report")

    scores_by_system = reveal_results(blind_batch, blind_scores)

    # Save per-system scores (matches existing report CLI glob pattern)
    for system_name, scores in scores_by_system.items():
        system_path = output_dir / f"scores_{system_name}.jsonl"
        save_judge_scores(scores, system_path)

    # Compile and save report
    report = compile_report(scores_by_system)
    md_path, json_path = save_report(report, output_dir)

    logger.info(
        "pipeline_complete",
        report_md=str(md_path),
        report_json=str(json_path),
        systems=list(scores_by_system.keys()),
    )

    return output_dir

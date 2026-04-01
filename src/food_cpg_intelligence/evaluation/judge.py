"""LLM-as-Judge evaluation module.

Scoring is designed to be executed by Claude Code agents (Opus 4.6 with
extended thinking) rather than via API calls. The module provides:
- Prompt templates for judge agents
- Score parsing from agent output files
- Batch orchestration helpers

The actual judging happens externally — either via spawned Claude Code
agents or via the Anthropic API as a fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from food_cpg_intelligence.evaluation.models import (
    EvalResponse,
    GoldStandardItem,
    JudgeScore,
)

logger = structlog.stdlib.get_logger(__name__)

JUDGE_DIMENSIONS = (
    "accuracy",
    "grounding",
    "completeness",
    "voice_fidelity",
    "hallucination_resistance",
)


def build_judge_prompt(
    question: str,
    reference_answer: str,
    response: str,
    *,
    system_label: str = "System",
) -> str:
    """Build the prompt for a judge agent to score a single response.

    The judge scores on 5 dimensions (1-5 scale) and provides reasoning.
    """
    return f"""You are an expert evaluator for a Canadian CPG (Consumer Packaged Goods) knowledge system called "Ask Peter". Your job is to score a system's response against a verified reference answer.

## Question
{question}

## Reference Answer (verified ground truth)
{reference_answer}

## {system_label}'s Response
{response}

## Scoring Instructions

Score the response on each dimension from 1 (worst) to 5 (best):

1. **Accuracy** (1-5): Is the information factually correct compared to the reference?
2. **Grounding** (1-5): Is the response grounded in real CPG knowledge (not generic filler)?
3. **Completeness** (1-5): Does it cover the key points from the reference answer?
4. **Voice Fidelity** (1-5): Does it sound like Peter Chapman — practical, specific, direct, Canadian CPG-focused?
5. **Hallucination Resistance** (1-5): Does it avoid stating things not supported by the reference? (5 = no hallucination)

## Output Format

You MUST respond with ONLY a JSON object (no markdown, no explanation outside the JSON):

```json
{{
  "accuracy": <1-5>,
  "grounding": <1-5>,
  "completeness": <1-5>,
  "voice_fidelity": <1-5>,
  "hallucination_resistance": <1-5>,
  "overall": <1-5>,
  "reasoning": "<2-3 sentences explaining the scores>"
}}
```"""


def parse_judge_output(raw: str) -> dict[str, float | str]:
    """Parse a judge's raw output into a scores dict.

    Handles JSON embedded in markdown code blocks or raw JSON.

    Raises:
        ValueError: If the output cannot be parsed as JSON.
    """
    text = raw.strip()
    # Strip markdown code blocks if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)

    try:
        parsed: dict[str, float | str] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {exc}\nRaw output: {raw[:200]}") from exc
    return parsed


def scores_dict_to_judge_score(
    scores: dict[str, float | str],
    *,
    question_id: str,
    system: str,
) -> JudgeScore:
    """Convert a parsed scores dict into a JudgeScore model."""
    return JudgeScore(
        question_id=question_id,
        system=system,
        accuracy=float(scores.get("accuracy", 0)),
        grounding=float(scores.get("grounding", 0)),
        completeness=float(scores.get("completeness", 0)),
        voice_fidelity=float(scores.get("voice_fidelity", 0)),
        hallucination_resistance=float(scores.get("hallucination_resistance", 0)),
        overall=float(scores.get("overall", 0)),
        reasoning=str(scores.get("reasoning", "")),
    )


def build_batch_judge_prompt(
    items: list[tuple[GoldStandardItem, EvalResponse]],
) -> str:
    """Build a prompt for judging multiple items in one agent call.

    Returns a prompt that asks the agent to score each item and write
    results to a JSONL format.
    """
    lines = [
        "You are an expert evaluator for the Ask Peter Canadian CPG knowledge system.",
        "Score each of the following question/response pairs on 5 dimensions (1-5 scale).",
        "",
        "For EACH item, output one line of JSON with these fields:",
        '{"question_id": "...", "accuracy": N, "grounding": N, "completeness": N, '
        '"voice_fidelity": N, "hallucination_resistance": N, "overall": N, '
        '"reasoning": "..."}',
        "",
        "Dimensions:",
        "- accuracy: factual correctness vs reference",
        "- grounding: grounded in real CPG knowledge",
        "- completeness: covers key points from reference",
        "- voice_fidelity: sounds like Peter Chapman (practical, specific, direct)",
        "- hallucination_resistance: avoids unsupported claims (5 = none)",
        "",
        f"--- {len(items)} ITEMS TO JUDGE ---",
        "",
    ]

    for gold, resp in items:
        lines.extend(
            [
                f"## Item: {gold.question_id}",
                f"**Question**: {gold.question}",
                f"**Reference Answer**: {gold.reference_answer}",
                f"**System Response**: {resp.response}",
                "",
            ]
        )

    lines.append("Output ONLY the JSONL lines, one per item. No other text.")
    return "\n".join(lines)


def parse_batch_judge_output(
    raw: str,
    *,
    system: str,
) -> list[JudgeScore]:
    """Parse multi-line JSONL output from a batch judge agent."""
    scores: list[JudgeScore] = []
    for raw_line in raw.strip().split("\n"):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            continue
        try:
            data = json.loads(stripped)
            if "question_id" in data:
                scores.append(
                    scores_dict_to_judge_score(
                        data,
                        question_id=data["question_id"],
                        system=system,
                    )
                )
        except json.JSONDecodeError:
            logger.warning("judge_parse_skip", line=stripped[:100])
    return scores


def save_judge_scores(scores: list[JudgeScore], path: Path) -> Path:
    """Save judge scores to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for score in scores:
            f.write(score.model_dump_json() + "\n")
    logger.info("saved_judge_scores", path=str(path), count=len(scores))
    return path


def load_judge_scores(path: Path) -> list[JudgeScore]:
    """Load judge scores from a JSONL file.

    Malformed lines are logged and skipped.
    """
    scores: list[JudgeScore] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, 1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                scores.append(JudgeScore.model_validate_json(stripped))
            except Exception as exc:
                logger.warning("judge_score_parse_error", line=line_num, error=str(exc))
    return scores

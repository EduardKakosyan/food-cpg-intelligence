"""Blind evaluation protocol — shuffle and mask system identity for fair judging."""

from __future__ import annotations

import random
import uuid

import structlog

from food_cpg_intelligence.evaluation.models import (
    BlindBatch,
    BlindItem,
    EvalResponse,
    GoldStandardItem,
    JudgeScore,
)


def prepare_blind_batch(
    gold_items: list[GoldStandardItem],
    responses_by_system: dict[str, list[EvalResponse]],
    *,
    seed: int = 42,
) -> BlindBatch:
    """Prepare a blinded batch for judge evaluation.

    Shuffles all responses across systems so the judge cannot identify
    which system produced which response.

    Args:
        gold_items: The gold standard items being evaluated.
        responses_by_system: Map of system name -> responses for each gold item.
        seed: Random seed for reproducible shuffling.

    Returns:
        BlindBatch with shuffled items and a hidden id-to-system mapping.
    """
    gold_by_qid = {item.question_id: item for item in gold_items}

    blind_items: list[BlindItem] = []
    id_to_system: dict[str, str] = {}

    for system_name, responses in responses_by_system.items():
        for resp in responses:
            gold = gold_by_qid.get(resp.question_id)
            if gold is None:
                continue

            blind_id = f"blind-{uuid.uuid4().hex[:8]}"
            id_to_system[blind_id] = system_name

            blind_items.append(
                BlindItem(
                    blind_id=blind_id,
                    question_id=resp.question_id,
                    question=gold.question,
                    reference_answer=gold.reference_answer,
                    response=resp.response,
                    category=gold.category,
                )
            )

    # Shuffle to eliminate ordering bias
    rng = random.Random(seed)
    rng.shuffle(blind_items)

    return BlindBatch(
        items=tuple(blind_items),
        id_to_system=id_to_system,
        shuffle_seed=seed,
    )


def reveal_results(
    batch: BlindBatch,
    scores: list[JudgeScore],
) -> dict[str, list[JudgeScore]]:
    """Unmask blinded scores back to their system identities.

    Args:
        batch: The original BlindBatch with the id-to-system mapping.
        scores: Judge scores keyed by blind_id (in question_id field).

    Returns:
        Dict mapping system name -> list of JudgeScores with real question_ids.
    """
    # Build lookup from blind_id to real question_id
    blind_to_question: dict[str, str] = {}
    for item in batch.items:
        blind_to_question[item.blind_id] = item.question_id

    logger = structlog.stdlib.get_logger(__name__)
    results: dict[str, list[JudgeScore]] = {}

    for score in scores:
        blind_id = score.question_id  # scores come in with blind_id as question_id
        system = batch.id_to_system.get(blind_id)
        if system is None:
            logger.warning("reveal_results_unknown_blind_id", blind_id=blind_id)
            continue
        real_qid = blind_to_question.get(blind_id, blind_id)

        unmasked = JudgeScore(
            question_id=real_qid,
            system=system,
            accuracy=score.accuracy,
            grounding=score.grounding,
            completeness=score.completeness,
            voice_fidelity=score.voice_fidelity,
            hallucination_resistance=score.hallucination_resistance,
            overall=score.overall,
            reasoning=score.reasoning,
        )

        if system not in results:
            results[system] = []
        results[system].append(unmasked)

    return results

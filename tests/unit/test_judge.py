"""Tests for LLM-as-judge module."""

from food_cpg_intelligence.evaluation.judge import (
    build_judge_prompt,
    parse_batch_judge_output,
    parse_judge_output,
    scores_dict_to_judge_score,
)


def test_build_judge_prompt_contains_fields() -> None:
    prompt = build_judge_prompt(
        question="What is X?",
        reference_answer="X is Y.",
        response="I think X is Y.",
    )
    assert "What is X?" in prompt
    assert "X is Y." in prompt
    assert "I think X is Y." in prompt
    assert "accuracy" in prompt.lower()
    assert "voice_fidelity" in prompt.lower()


def test_parse_judge_output_raw_json() -> None:
    raw = '{"accuracy": 4, "grounding": 3, "completeness": 5, "voice_fidelity": 3, "hallucination_resistance": 4, "overall": 4, "reasoning": "Good."}'
    scores = parse_judge_output(raw)
    assert scores["accuracy"] == 4
    assert scores["overall"] == 4


def test_parse_judge_output_with_code_block() -> None:
    raw = """Here are the scores:
```json
{"accuracy": 5, "grounding": 4, "completeness": 4, "voice_fidelity": 3, "hallucination_resistance": 5, "overall": 4, "reasoning": "Excellent."}
```"""
    scores = parse_judge_output(raw)
    assert scores["accuracy"] == 5


def test_scores_dict_to_judge_score() -> None:
    scores = {
        "accuracy": 4.0,
        "grounding": 3.5,
        "completeness": 4.0,
        "voice_fidelity": 3.0,
        "hallucination_resistance": 5.0,
        "overall": 4.0,
        "reasoning": "Good response.",
    }
    judge_score = scores_dict_to_judge_score(
        scores,
        question_id="q1",
        system="rag",
    )
    assert judge_score.question_id == "q1"
    assert judge_score.system == "rag"
    assert judge_score.accuracy == 4.0
    assert judge_score.hallucination_resistance == 5.0


def test_parse_batch_judge_output() -> None:
    raw = """{"question_id": "q1", "accuracy": 4, "grounding": 3, "completeness": 4, "voice_fidelity": 3, "hallucination_resistance": 5, "overall": 4, "reasoning": "Good."}
{"question_id": "q2", "accuracy": 2, "grounding": 2, "completeness": 2, "voice_fidelity": 2, "hallucination_resistance": 3, "overall": 2, "reasoning": "Weak."}"""
    scores = parse_batch_judge_output(raw, system="finetune")
    assert len(scores) == 2
    assert scores[0].question_id == "q1"
    assert scores[1].overall == 2.0


def test_parse_batch_judge_output_skips_bad_lines() -> None:
    raw = """{"question_id": "q1", "accuracy": 4, "grounding": 3, "completeness": 4, "voice_fidelity": 3, "hallucination_resistance": 5, "overall": 4, "reasoning": "OK."}
not valid json
```
{"question_id": "q2", "accuracy": 3, "grounding": 3, "completeness": 3, "voice_fidelity": 3, "hallucination_resistance": 3, "overall": 3, "reasoning": "Fair."}"""
    scores = parse_batch_judge_output(raw, system="rag")
    assert len(scores) == 2

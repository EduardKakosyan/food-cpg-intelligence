"""FAQ smoke test for the voice-v1 adapter.

Loads Qwen3.5-9B + the LoRA adapter from Run 003 and generates responses to
the 82 Peter-validated FAQ questions. Writes a JSONL of (question, response)
pairs for human review of voice quality.

Usage: uv run python scripts/faq_smoke_test.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are Ask Peter -- an AI assistant that channels Peter Chapman's decades "
    "of Canadian CPG expertise. You give practical, specific advice to food and "
    "beverage SMEs on retailer meetings, pricing strategy, trade shows, "
    "promotion planning, and category management. Ground your answers in real "
    "industry knowledge and be direct."
)


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
            if limit and len(questions) >= limit:
                break
    return questions


def _format_chatml(
    tokenizer, system: str, user: str, *, skip_thinking: bool = True
) -> torch.Tensor:
    """Apply ChatML chat template the way Qwen 3.5 expects.

    Qwen 3.5 base has a built-in reasoning mode (<think>...</think>) that fires
    by default. Our training data has no think tags, so the LoRA can't fully
    suppress them — the model burns the entire token budget on meta-analysis.
    Pre-filling `</think>\\n\\n` after the assistant turn forces the model to
    skip thinking and produce the actual answer directly.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if skip_thinking:
        prompt_text = prompt_text + "</think>\n\n"
    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="models/adapters/skufood-voice-v1",
        help="Adapter dir.",
    )
    parser.add_argument(
        "--base",
        default="Qwen/Qwen3.5-9B",
        help="Base model (defaults to local HF cache).",
    )
    parser.add_argument(
        "--faq",
        default="data/training/sft_mvp/inference_test_faq.jsonl",
        help="FAQ JSONL.",
    )
    parser.add_argument(
        "--out",
        default="data/evaluation/faq_responses_voice-v1.jsonl",
        help="Output JSONL.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Truncate FAQ count.")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default=None, help="Override device.")
    args = parser.parse_args()

    device = args.device or _select_device()
    print(f"Device: {device}")

    adapter_dir = Path(args.adapter)
    faq_path = Path(args.faq)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        print(f"ERROR: adapter not found at {adapter_dir}")
        sys.exit(1)
    if not faq_path.exists():
        print(f"ERROR: FAQ not found at {faq_path}")
        sys.exit(1)

    questions = _load_questions(faq_path, args.limit)
    print(f"Loaded {len(questions)} FAQ questions")

    # Tokenizer from the adapter dir (has the ChatML chat_template).
    print(f"Loading tokenizer from {adapter_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model in fp16. ~18 GB; M4 Pro 48GB unified memory handles it.
    print(f"Loading {args.base} (fp16, may take a minute)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"Loading adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    # Resolve EOS for ChatML — Run 001 retrospective: must set explicitly.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id == tokenizer.unk_token_id or im_end_id is None:
        print("WARNING: <|im_end|> not in vocab; relying on default EOS.")
        im_end_id = tokenizer.eos_token_id

    # Append-as-we-go so a mid-run crash (MPS OOM, OS swap) doesn't lose work.
    # Truncate the output file at start so we don't accumulate stale runs.
    out_path.write_text("", encoding="utf-8")
    written = 0
    start = time.time()
    with out_path.open("a", encoding="utf-8") as out_f:
        for i, q in enumerate(questions, start=1):
            question = q["instruction"]
            context = q.get("context") or ""
            user_msg = f"{question}\n\nContext:\n{context}" if context else question

            input_ids = _format_chatml(tokenizer, SYSTEM_PROMPT, user_msg).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.9,
                    eos_token_id=im_end_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = output_ids[0][input_ids.shape[1] :]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            record = {
                "test_id": q.get("test_id", f"faq-{i:03d}"),
                "question": question,
                "context": context,
                "response": response,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1

            elapsed = time.time() - start
            avg = elapsed / i
            eta = avg * (len(questions) - i)
            print(
                f"[{i:3d}/{len(questions)}] {elapsed:6.1f}s elapsed, ~{eta:6.1f}s left "
                f"({avg:.1f}s/q) — {question[:60]}"
            )

    print(f"\nWrote {written} responses to {out_path}")


if __name__ == "__main__":
    main()

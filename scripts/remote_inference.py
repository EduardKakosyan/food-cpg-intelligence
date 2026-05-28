"""Post-training FAQ inference on Lambda.

Loads the merged 16-bit model that the training script just wrote and runs
inference on the 82 Peter-validated FAQ questions. Designed to run on the
same Lambda instance immediately after training so the artifacts are still
on disk and the GPU is warm.

Outputs `data/evaluation/faq_responses_<tag>.jsonl` — pull via scp before
auto-shutdown.

Usage: python scripts/remote_inference.py --config configs/training_gpu.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import yaml

warnings.filterwarnings("ignore", message=".*_check_is_size.*", category=FutureWarning)

# Qwen 3.5 has a default reasoning mode that fires <think>...</think> before
# the actual answer. The training data has no think tags, so the LoRA can't
# fully suppress this — pre-filling </think>\n\n after the assistant turn
# tells the model "skip thinking, answer directly". We also explicitly tell
# the model to avoid markdown formatting since Peter writes in flowing prose.
SYSTEM_PROMPT = (
    "You are Ask Peter -- an AI assistant that channels Peter Chapman's decades "
    "of Canadian CPG expertise. You give practical, specific advice to food and "
    "beverage SMEs on retailer meetings, pricing strategy, trade shows, "
    "promotion planning, and category management. Ground your answers in real "
    "industry knowledge and be direct.\n\n"
    "Format: write in flowing prose like Peter does in his newsletters. Do not "
    "use markdown headers (#, ##), bullet points, bold/italics, or numbered "
    "lists. Speak in the first person."
)


def _load_questions(path: Path, limit: int | None) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            questions.append(json.loads(stripped))
            if limit and len(questions) >= limit:
                break
    return questions


def _format_chatml(tokenizer, system: str, user: str, *, skip_thinking: bool):
    """Apply ChatML, optionally pre-filling </think> to skip reasoning mode."""
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if skip_thinking:
        prompt_text += "</think>\n\n"
    return tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-training FAQ inference")
    parser.add_argument("--config", default="configs/training_gpu.yaml")
    parser.add_argument(
        "--faq",
        default="data/training/sft_mvp/inference_test_faq.jsonl",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSONL. Defaults to data/evaluation/faq_responses_<tag>.jsonl "
        "where <tag> is derived from merged_dir basename.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--skip-thinking", action="store_true", default=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    merged_dir = Path(config.get("merged_dir", "models/merged/skufood"))
    if not merged_dir.exists():
        print(f"ERROR: merged model not found at {merged_dir}")
        print("(Run remote_train_unsloth.py first; GGUF fallback writes merged.)")
        sys.exit(1)

    tag = merged_dir.name
    out_path = Path(args.out) if args.out else Path(f"data/evaluation/faq_responses_{tag}.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")

    faq_path = Path(args.faq)
    if not faq_path.exists():
        print(f"ERROR: FAQ not found at {faq_path}")
        sys.exit(1)

    questions = _load_questions(faq_path, args.limit)
    print(f"Loaded {len(questions)} FAQ questions")

    # Lazy-import heavy ML deps — keeps argparse + dry-run paths fast.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading merged model from {merged_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id == tokenizer.unk_token_id or im_end_id is None:
        im_end_id = tokenizer.eos_token_id

    written = 0
    start = time.time()
    with out_path.open("a", encoding="utf-8") as out_f:
        for i, q in enumerate(questions, start=1):
            question = q["instruction"]
            context = q.get("context") or ""
            user_msg = f"{question}\n\nContext:\n{context}" if context else question

            encoded = _format_chatml(
                tokenizer, SYSTEM_PROMPT, user_msg, skip_thinking=args.skip_thinking
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
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
                "model": tag,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1

            elapsed = time.time() - start
            avg = elapsed / i
            eta = avg * (len(questions) - i)
            print(
                f"[{i:3d}/{len(questions)}] {elapsed:6.1f}s elapsed, "
                f"~{eta:6.1f}s left ({avg:.1f}s/q) — {question[:55]}",
                flush=True,
            )

    print(f"\nWrote {written} responses to {out_path}")


if __name__ == "__main__":
    main()

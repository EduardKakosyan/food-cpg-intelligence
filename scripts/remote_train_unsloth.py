"""Unsloth QLoRA training script for remote GPU (Lambda.ai / RunPod).

Based on Unsloth's official Qwen fine-tuning examples:
https://unsloth.ai/docs/models/qwen3.5/fine-tune

Usage:
    python scripts/remote_train_unsloth.py --config configs/training_gpu.yaml
    python scripts/remote_train_unsloth.py --dry-run  # validate config only (no unsloth)

The dry-run path is intentionally kept free of unsloth/trl imports so you can
validate the YAML + data wiring locally on machines without CUDA (M-series Macs,
CI runners). The heavy imports happen inside `_train()` and only on the GPU.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import yaml

# bitsandbytes emits a deprecation warning from a torch internal that's harmless;
# silence it so the dry-run output is clean. Applies to real training too.
warnings.filterwarnings("ignore", message=".*_check_is_size.*", category=FutureWarning)


def _load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
    return result


def _print_config_summary(config: dict[str, Any]) -> None:
    """Echo the loaded config for run reproducibility / dry-run output."""
    lora = config.get("lora", {})
    print(f"Model: {config.get('model')}")
    print(f"LoRA rank: {lora.get('rank')}, alpha: {lora.get('alpha')}")
    print(
        f"LoRA target_modules: "
        f"{lora.get('target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])}"
    )
    print(f"Batch size: {config.get('batch_size')}, Grad accum: {config.get('grad_accumulation')}")
    print(f"Epochs: {config.get('num_train_epochs', 3)}")
    print(f"Learning rate: {config.get('learning_rate', 2e-4)}")
    print(f"NEFTune alpha: {config.get('neftune_noise_alpha')}")


def _validate_data_paths(config: dict[str, Any]) -> tuple[Path, Path | None]:
    """Confirm train.jsonl exists at the configured path; return (train, val)."""
    data_dir = Path(config.get("data_dir", "data/training/formatted"))
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    if not train_file.exists():
        print(f"ERROR: Training data not found: {train_file}")
        sys.exit(1)

    train_count = sum(1 for _ in train_file.open())
    val_count = sum(1 for _ in val_file.open()) if val_file.exists() else 0
    print(f"Data: {train_count} train, {val_count} val ({data_dir})")
    return train_file, val_file if val_file.exists() else None


def _train(config: dict[str, Any], train_file: Path, val_file: Path | None) -> None:
    """Actual training — imports unsloth/trl, requires CUDA. Never runs in dry-run."""
    # Import order matters: unsloth must come before trl / transformers / peft.
    import json

    import unsloth  # noqa: F401  (must precede TRL imports for optimizations)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template

    max_seq_length = config.get("max_seq_length", 2048)

    print(f"\nLoading {config['model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"],
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    # Maps <|im_end|> to EOS — critical per Run 001 retrospective.
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")

    lora_cfg = config["lora"]
    # Default: all linear (Run 001/002). Voice transfer (Run 003+) uses
    # attention-only via configs/training_gpu.yaml -> lora.target_modules.
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    print(f"LoRA target_modules: {target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=0,  # must be 0 for Unsloth fast patching
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    def load_jsonl(path: Path) -> Dataset:
        data = [json.loads(line.strip()) for line in path.open() if line.strip()]
        return Dataset.from_list(data)

    train_dataset = load_jsonl(train_file)
    val_dataset = load_jsonl(val_file) if val_file else None
    print(f"Loaded: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val")

    output_dir = config.get("output_dir", "models/adapters/skufood")

    sft_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("grad_accumulation", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type="cosine",
        warmup_steps=config.get("warmup_steps", 50),
        weight_decay=config.get("weight_decay", 0.01),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.get("eval_steps", 100) if val_dataset else None,
        save_total_limit=5,
        optim="adamw_8bit",
        report_to="none",
        seed=config.get("seed", 42),
        # NEFTune alpha — input-embedding noise for style transfer
        # (Jain et al., arXiv:2310.05914). Only applies when value is non-null.
        neftune_noise_alpha=config.get("neftune_noise_alpha"),
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_args,
    )

    print("\nStarting training...")
    trainer.train()

    print(f"\nSaving adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # GGUF must happen on GPU; fall back to merged 16-bit if quantize hangs.
    gguf_dir = config.get("gguf_dir", "models/gguf")
    print(f"Saving GGUF to {gguf_dir}... (this installs llama.cpp if needed)")
    import subprocess

    subprocess.run(
        ["sudo", "apt-get", "install", "-y", "libssl-dev", "libcurl4-openssl-dev"],
        capture_output=True,
    )
    try:
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        print(f"GGUF saved to {gguf_dir}/")
    except Exception as e:
        print(f"GGUF export failed: {e}")
        print("Falling back to merged safetensors export...")
        merged_dir = config.get("merged_dir", "models/merged/skufood")
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print(f"Merged model saved to {merged_dir}/")

    print("\nTraining complete!")
    print("\nDownload your model:")
    print(f"  scp -r user@<IP>:{gguf_dir}/ .")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth QLoRA fine-tuning")
    parser.add_argument(
        "--config",
        default="configs/training_gpu.yaml",
        help="Training config YAML path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + data without loading unsloth/CUDA. Safe to run on Mac.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    _print_config_summary(config)

    train_file, val_file = _validate_data_paths(config)

    if args.dry_run:
        print("\nDry run — config and data validated (no unsloth imported).")
        return

    _train(config, train_file, val_file)


if __name__ == "__main__":
    main()

"""Unsloth QLoRA training script for remote GPU (Lambda.ai / RunPod).

Usage:
    python scripts/remote_train_unsloth.py --config configs/training_gpu.yaml
    python scripts/remote_train_unsloth.py  # uses default config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth QLoRA fine-tuning")
    parser.add_argument(
        "--config",
        default="configs/training_gpu.yaml",
        help="Training config YAML path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Model: {config['model']}")
    print(f"LoRA rank: {config['lora']['rank']}, alpha: {config['lora']['alpha']}")
    print(f"Batch size: {config['batch_size']}, Grad accum: {config['grad_accumulation']}")
    print(f"Epochs: {config.get('num_train_epochs', 3)}")

    # Check data exists
    data_dir = Path(config.get("data_dir", "data/training/formatted"))
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"

    if not train_file.exists():
        print(f"ERROR: Training data not found: {train_file}")
        sys.exit(1)

    train_count = sum(1 for _ in train_file.open())
    val_count = sum(1 for _ in val_file.open()) if val_file.exists() else 0
    print(f"Data: {train_count} train, {val_count} val")

    if args.dry_run:
        print("Dry run — config and data validated.")
        return

    # Import unsloth FIRST — must come before transformers/peft/trl
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    # Load model
    print(f"\nLoading {config['model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"],
        max_seq_length=config.get("max_seq_length", 2048),
        load_in_4bit=True,
    )

    # Apply LoRA
    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load dataset
    def load_jsonl(path: Path) -> Dataset:
        data = []
        with path.open() as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return Dataset.from_list(data)

    train_dataset = load_jsonl(train_file)
    val_dataset = load_jsonl(val_file) if val_file.exists() else None

    print(f"Loaded: {len(train_dataset)} train, {len(val_dataset) if val_dataset else 0} val")

    # Training config — uses SFTConfig (not TrainingArguments)
    output_dir = config.get("output_dir", "models/adapters/skufood")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("grad_accumulation", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type="cosine",
        warmup_steps=config.get("warmup_steps", 50),
        weight_decay=config.get("weight_decay", 0.01),
        fp16=not config.get("bf16", True),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.get("eval_steps", 100) if val_dataset else None,
        save_total_limit=5,
        report_to="none",
        seed=config.get("seed", 42),
        max_seq_length=config.get("max_seq_length", 2048),
    )

    # Train — current TRL API: model + args + dataset + peft_config
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save merged model for easy export
    merged_dir = config.get("merged_dir", "models/merged/skufood")
    print(f"Saving merged model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    # Also save GGUF if requested
    if config.get("save_gguf", False):
        gguf_dir = config.get("gguf_dir", "models/gguf")
        print(f"Saving GGUF to {gguf_dir}...")
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()

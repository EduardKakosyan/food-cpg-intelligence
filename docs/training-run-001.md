# Training Run 001 — SKUFood Ask Peter Fine-Tune

**Date:** 2026-04-22
**Status:** Complete

## Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen3.5-9B (multimodal) |
| Method | QLoRA (4-bit) via Unsloth |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Trainable params | 58.2M / 9.47B (0.61%) |
| Batch size | 4 (effective 16 with grad_accum=4) |
| Epochs | 3 |
| Learning rate | 2e-4 (cosine schedule) |
| Warmup steps | 50 |
| Max seq length | 2048 |
| Optimizer | adamw_8bit |
| Precision | bf16 |

## Infrastructure

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| Provider | Lambda.ai |
| CUDA | 8.0 / Toolkit 13.0 |
| Training time | 37 minutes (345 steps @ ~6s/step) |
| GGUF export | Separate step via llama.cpp (not unsloth's built-in) |

## Dataset

| Split | Samples |
|-------|---------|
| Train | 1,828 |
| Val | 204 |
| Format | ChatML (`<\|im_start\|>` / `<\|im_end\|>`) |
| Avg response length | 106 words (P95: 280, max: 338) |

## Loss Curve

| Epoch | Train Loss | Eval Loss | Grad Norm |
|-------|-----------|-----------|-----------|
| 0.09 | 2.886 | - | 1.322 |
| 0.26 | 1.594 | - | 0.609 |
| 0.53 | 1.358 | - | 0.591 |
| 0.88 | 1.273 | **1.274** | 0.504 |
| 1.13 | 1.162 | - | 0.535 |
| 1.48 | 1.119 | - | 0.675 |
| 1.74 | 1.090 | **1.205** | 0.669 |
| 2.00 | 1.073 | - | 1.268 |
| 2.35 | 0.797 | - | 0.918 |
| 2.61 | 0.830 | **1.227** | 0.863 |
| 2.88 | 0.759 | - | 0.931 |
| 3.00 | - | **1.221** | - |

**Final train loss:** 1.157 (avg), 0.766 (last logged step)
**Final eval loss:** 1.221

Eval loss plateaued at ~1.2 from epoch 1.7 onward while train loss kept dropping to 0.76 — mild overfitting in epoch 3. Consider 2 epochs for next run.

## Eval Results (10 val samples, GPU inference, 4-bit)

Model correctly learned:
- Peter Chapman's voice and tone ("The first thing you need to be is well prepared...")
- Domain-specific CPG terminology (retailer relationships, cost to retailer, pricing, category management)
- Practical advice framing (actionable, direct, grounded)

Issues observed:
- **Boundary questions not enforced:** Sample 1 asked about Costco's Kirkland brand (expected: "not in my wheelhouse"). Model answered anyway with fabricated details.
- **Generalization over memorization:** Model captures the style and gives relevant advice but generates its own continuations rather than reproducing source material verbatim. This is expected and desirable for a Q&A assistant.
- **EOS token handling:** Inference required explicit `eos_token_id=<|im_end|>` — model would generate endlessly without it.

## Outputs

| Artifact | Location | Size |
|----------|----------|------|
| LoRA adapter | `models/adapters/skufood/` (remote) | 223 MB |
| Merged 16-bit | `models/merged/skufood/` (remote) | 18 GB |
| GGUF Q4_K_M | `models/gguf/skufood-9b-q4_k_m.gguf` (local + remote) | 5.3 GB |
| Checkpoints | `models/adapters/skufood/checkpoint-{100,200,300,345}` (remote) | - |

## Lessons Learned for Next Run

### 1. Model selection: use text-only, not multimodal
Qwen3.5-9B is a vision-language model. For text-only fine-tuning:
- The multimodal processor (`Qwen2VLProcessor`) wraps the tokenizer and tries to parse inputs as images during inference
- Requires using `tokenizer.tokenizer` (inner text tokenizer) to bypass image processing
- `apply_chat_template()` fails because it expects multimodal content format

**Next run:** Use a text-only model like **Qwen3-8B**, **Qwen2.5-7B-Instruct**, or **Mistral-7B-Instruct-v0.3**.

### 2. Dependency management: never upgrade torch/transformers independently
Unsloth pins strict version ranges (`transformers<=5.5.0`, `torch<2.11.0`, `datasets<4.4.0`). Running `pip install --upgrade transformers` installs versions above these caps, causing:
- `KeyError: 'qwen3_5'` (too old)
- Version conflict warnings (too new)
- TRL API breakage (`tokenizer` renamed to `processing_class`, kwargs moved to `SFTConfig`)

**Next run:** Let `pip install unsloth[...]` manage torch/transformers/datasets. Only install non-overlapping packages separately.

### 3. Import order matters
Unsloth must be imported **before** trl, transformers, and peft at the module level. Importing inside a function after other ML imports triggers a warning and may miss optimizations.

### 4. GGUF export: use llama.cpp directly, not unsloth's built-in
Unsloth's `save_pretrained_gguf()` tries to `sudo apt-get install` interactively, which hangs in non-interactive SSH. The manual pipeline works reliably:
```bash
# 1. Convert HF safetensors to GGUF f16
python llama.cpp/convert_hf_to_gguf.py models/merged/skufood/ --outfile models/gguf/model-f16.gguf --outtype f16

# 2. Quantize f16 to Q4_K_M
llama.cpp/build/bin/llama-quantize models/gguf/model-f16.gguf models/gguf/model-q4_k_m.gguf Q4_K_M
```
Note: `convert_hf_to_gguf.py --outtype` only accepts base types (f32, f16, bf16, q8_0) — quantization is a separate step.

### 5. EOS token must be set explicitly for inference
Without `eos_token_id=<|im_end|>` in `model.generate()`, the model generates indefinitely. The training script sets EOS correctly during training, but inference scripts must set it manually.

### 6. Docker-first deployment
Never live-debug dependencies on a paid GPU instance. Build and validate the Docker image locally, then ship the frozen image:
```bash
bash scripts/remote_train.sh --package-docker  # build + validate locally
scp skufood-train.tar.gz ubuntu@<IP>:~/         # upload frozen image
ssh ubuntu@<IP> 'docker load < skufood-train.tar.gz && bash scripts/remote_train.sh --docker'
```

### 7. Epochs: 2 may be better than 3
Eval loss plateaued at 1.205 by epoch 1.7 and barely improved to 1.221 by epoch 3, while train loss dropped from 1.09 to 0.76 — classic overfitting signal. Try 2 epochs next time.

### 8. Boundary enforcement needs training data work
The model doesn't refuse out-of-scope questions. The training data needs more examples of boundary responses ("that's not in my wheelhouse") to teach the model when NOT to answer.

## Next Steps

1. Test GGUF locally via Ollama on Mac M4 Pro
2. Run full eval suite (RAGAS + LLM-as-judge) against Claude+RAG baseline
3. Consider retraining with text-only model (Qwen3-8B) for cleaner inference
4. Add more boundary/refusal examples to training data
5. Try 2 epochs instead of 3

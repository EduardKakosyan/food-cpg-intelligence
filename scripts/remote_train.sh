#!/bin/bash
# Remote GPU training script for Lambda.ai / RunPod / Vast.ai
#
# Quick start (on the remote GPU instance):
#   1. Upload training data: scp -r data/training/formatted user@gpu-host:/workspace/data/training/
#   2. SSH into the instance
#   3. Run: bash scripts/remote_train.sh
#
# Or use the package command to create a portable tarball:
#   bash scripts/remote_train.sh --package
#   # Creates skufood-training.tar.gz — upload to GPU instance and extract

set -euo pipefail

if [ "${1:-}" = "--package" ]; then
    echo "=== Packaging training data + scripts for upload ==="
    tar czf skufood-training.tar.gz \
        configs/training_gpu.yaml \
        scripts/remote_train_unsloth.py \
        scripts/requirements-gpu.txt \
        scripts/remote_train.sh \
        data/training/formatted/train.jsonl \
        data/training/formatted/val.jsonl \
        data/training/formatted/dataset_meta.json \
        2>/dev/null

    SIZE=$(du -h skufood-training.tar.gz | cut -f1)
    echo "Created: skufood-training.tar.gz ($SIZE)"
    echo ""
    echo "Upload to your GPU instance and run:"
    echo "  tar xzf skufood-training.tar.gz"
    echo "  pip install -r scripts/requirements-gpu.txt"
    echo "  python scripts/remote_train_unsloth.py"
    exit 0
fi

echo "=== SKUFood Remote GPU Training ==="
echo ""

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Are you on a GPU instance?"
fi
nvidia-smi 2>/dev/null || true

# Install deps if not already
if ! python -c "import unsloth" 2>/dev/null; then
    echo ""
    echo "Installing dependencies..."
    pip install -r scripts/requirements-gpu.txt
fi

# Validate
echo ""
echo "Validating config and data..."
python scripts/remote_train_unsloth.py --dry-run

# Train
echo ""
echo "Starting training..."
python scripts/remote_train_unsloth.py --config configs/training_gpu.yaml

echo ""
echo "=== Training complete ==="
echo "Adapter: models/adapters/skufood/"
echo "Merged:  models/merged/skufood/"
echo "GGUF:    models/gguf/"
echo ""
echo "Download the GGUF for Ollama:"
echo "  scp gpu-host:/workspace/models/gguf/*.gguf ."

# Safety: auto-shutdown after training to prevent runaway billing
if [ "${AUTO_SHUTDOWN:-true}" = "true" ]; then
    echo ""
    echo "!!! AUTO-SHUTDOWN in 10 minutes to prevent billing overrun !!!"
    echo "!!! Cancel with: sudo shutdown -c                          !!!"
    echo "!!! Download your models before then!                      !!!"
    sudo shutdown -h +10 "Training complete — auto-shutdown to prevent billing"
fi

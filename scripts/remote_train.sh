#!/bin/bash
# Remote GPU training script for Lambda.ai / RunPod / Vast.ai
#
# Quick start (on the remote GPU instance):
#   1. Upload training data: scp -r data/training/formatted user@gpu-host:/home/ubuntu/
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
        scripts/remote_train.sh \
        data/training/formatted/train.jsonl \
        data/training/formatted/val.jsonl \
        data/training/formatted/dataset_meta.json \
        2>/dev/null

    SIZE=$(du -h skufood-training.tar.gz | cut -f1)
    echo "Created: skufood-training.tar.gz ($SIZE)"
    echo ""
    echo "Upload to your GPU instance and run:"
    echo "  scp skufood-training.tar.gz ubuntu@<IP>:/home/ubuntu/"
    echo "  ssh ubuntu@<IP>"
    echo "  tar xzf skufood-training.tar.gz && bash scripts/remote_train.sh"
    exit 0
fi

echo "=== SKUFood Remote GPU Training ==="
echo ""

# ──── Step 1: Check GPU ────
echo "Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This script requires an NVIDIA GPU instance."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ──── Step 2: Create clean venv (avoids ALL system package conflicts) ────
VENV_DIR="$HOME/skufood-venv"
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating clean Python venv at $VENV_DIR..."
    python3 -m venv "$VENV_DIR" --clear
fi
source "$VENV_DIR/bin/activate"
echo "Using venv: $(which python) ($(python --version))"
echo ""

# ──── Step 3: Install ALL dependencies in one shot ────
# Check if unsloth is already installed
if ! python -c "import unsloth" 2>/dev/null; then
    echo "Installing dependencies (this takes 3-5 minutes)..."
    echo ""

    # Unsloth's official install — handles torch, triton, xformers
    pip install --upgrade pip
    pip install "unsloth[cu124-ampere] @ git+https://github.com/unslothai/unsloth.git"

    # Everything else the training script needs
    pip install \
        transformers \
        peft \
        trl \
        accelerate \
        bitsandbytes \
        datasets \
        pyyaml \
        scipy \
        scikit-learn \
        unsloth-zoo \
        torchvision

    echo ""
    echo "Dependencies installed."
else
    echo "Dependencies already installed, skipping."
fi
echo ""

# ──── Step 4: Validate ────
echo "Validating config and data..."
python scripts/remote_train_unsloth.py --dry-run
echo ""

# ──── Step 5: Train ────
echo "Starting training..."
python scripts/remote_train_unsloth.py --config configs/training_gpu.yaml

echo ""
echo "=== Training complete ==="
echo "Adapter: models/adapters/skufood/"
echo "Merged:  models/merged/skufood/"
echo "GGUF:    models/gguf/"
echo ""
echo "Download the GGUF for Ollama:"
echo "  scp -i <key.pem> ubuntu@<IP>:/home/ubuntu/models/gguf/*.gguf ."

# Safety: auto-shutdown after training to prevent runaway billing
if [ "${AUTO_SHUTDOWN:-true}" = "true" ]; then
    echo ""
    echo "!!! AUTO-SHUTDOWN in 10 minutes to prevent billing overrun !!!"
    echo "!!! Cancel with: sudo shutdown -c                          !!!"
    echo "!!! Download your models before then!                      !!!"
    sudo shutdown -h +10 "Training complete — auto-shutdown to prevent billing"
fi

#!/bin/bash
# SKUFood QLoRA Training — Docker-first, bare-metal fallback
#
# RECOMMENDED (Docker — reproducible, tested locally before deploying):
#   bash scripts/remote_train.sh --package-docker   # build image + export as tarball
#   scp skufood-train.tar.gz ubuntu@<IP>:~/
#   ssh ubuntu@<IP> 'docker load < skufood-train.tar.gz && bash scripts/remote_train.sh --docker'
#
# Or on any machine with docker + nvidia-container-toolkit:
#   bash scripts/remote_train.sh --docker            # run via Docker
#   bash scripts/remote_train.sh --docker --dry-run   # validate only
#
# Legacy bare-metal (NOT recommended):
#   bash scripts/remote_train.sh --bare
#
# Package data + scripts for upload (bare-metal):
#   bash scripts/remote_train.sh --package

set -euo pipefail

MODE="${1:-}"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --package: Create tarball for bare-metal remote upload
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$MODE" = "--package" ]; then
    # Path comes from configs/training_gpu.yaml — keep in sync if you change it.
    DATA_DIR="${DATA_DIR:-data/training/sft_mvp}"
    echo "=== Packaging training data ($DATA_DIR) + scripts for bare-metal upload ==="
    tar czf skufood-training.tar.gz \
        configs/training_gpu.yaml \
        scripts/remote_train_unsloth.py \
        scripts/remote_train.sh \
        "$DATA_DIR/train.jsonl" \
        "$DATA_DIR/val.jsonl" \
        "$DATA_DIR/dataset_meta.json" \
        2>/dev/null

    SIZE=$(du -h skufood-training.tar.gz | cut -f1)
    echo "Created: skufood-training.tar.gz ($SIZE)"
    echo ""
    echo "Upload to your GPU instance and run:"
    echo "  scp skufood-training.tar.gz ubuntu@<IP>:/home/ubuntu/"
    echo "  ssh ubuntu@<IP>"
    echo "  tar xzf skufood-training.tar.gz && bash scripts/remote_train.sh --bare"
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --package-docker: Build image + export as a loadable tarball
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$MODE" = "--package-docker" ]; then
    echo "=== Building Docker image ==="
    docker build -t skufood-train -f docker/training/Dockerfile .

    echo ""
    echo "=== Validating image (dry-run) ==="
    docker run --rm skufood-train --dry-run

    echo ""
    echo "=== Exporting image ==="
    docker save skufood-train | gzip > skufood-train.tar.gz
    SIZE=$(du -h skufood-train.tar.gz | cut -f1)
    echo "Created: skufood-train.tar.gz ($SIZE)"
    echo ""
    echo "Upload to your GPU instance and run:"
    echo "  scp skufood-train.tar.gz ubuntu@<IP>:/home/ubuntu/"
    echo "  ssh ubuntu@<IP>"
    echo "  docker load < skufood-train.tar.gz"
    echo "  bash scripts/remote_train.sh --docker"
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --docker: Run training inside container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$MODE" = "--docker" ]; then
    echo "=== SKUFood Docker GPU Training ==="
    shift  # consume --docker, pass remaining args (e.g. --dry-run) to container

    # Ensure nvidia-container-toolkit is working
    if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo "ERROR: nvidia-container-toolkit not working. Install it:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi

    # mkdir is defensive — docker -v will create missing dirs anyway. The actual
    # data path is read from configs/training_gpu.yaml at runtime.
    mkdir -p models data/training/sft_mvp

    docker run --rm \
        --gpus all \
        --shm-size=16g \
        -v "$(pwd)/data:/workspace/data" \
        -v "$(pwd)/models:/workspace/models" \
        -v "$(pwd)/configs:/workspace/configs" \
        skufood-train "$@"

    echo ""
    echo "=== Training complete ==="
    echo "Models saved to: ./models/"

    # Auto-shutdown if on a remote GPU instance
    if [ "${AUTO_SHUTDOWN:-false}" = "true" ]; then
        echo ""
        echo "!!! AUTO-SHUTDOWN in 10 minutes to prevent billing overrun !!!"
        echo "!!! Cancel with: sudo shutdown -c                          !!!"
        sudo shutdown -h +10 "Training complete — auto-shutdown to prevent billing"
    fi
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# --bare: Legacy bare-metal install (fallback for instances without Docker)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$MODE" = "--bare" ]; then
    echo "=== SKUFood Bare-Metal GPU Training ==="
    echo "WARNING: Prefer --docker for reproducible builds."
    echo ""

    # Check GPU
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. This script requires an NVIDIA GPU instance."
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""

    # Create venv
    VENV_DIR="$HOME/skufood-venv"
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "Creating clean Python venv at $VENV_DIR..."
        python3 -m venv "$VENV_DIR" --clear
    fi
    source "$VENV_DIR/bin/activate"
    echo "Using venv: $(which python) ($(python --version))"
    echo ""

    # Install deps — unsloth manages torch/transformers/datasets versions.
    # DO NOT pip install --upgrade transformers/datasets/torch separately!
    echo "Installing/upgrading dependencies..."
    pip install --upgrade pip
    pip install --upgrade "unsloth @ git+https://github.com/unslothai/unsloth.git"
    pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth_zoo
    pip install \
        peft \
        trl \
        accelerate \
        bitsandbytes \
        pyyaml \
        scipy \
        scikit-learn \
        torchvision
    # Flash attention + fast path deps (optional — don't fail if build breaks)
    pip install --no-build-isolation flash-attn causal-conv1d 2>/dev/null || echo "flash-attn build failed (optional, continuing)"
    pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git" 2>/dev/null || echo "flash-linear-attention failed (optional, continuing)"
    echo "Dependencies installed."
    echo ""

    # Validate
    python scripts/remote_train_unsloth.py --dry-run
    echo ""

    # Train
    python scripts/remote_train_unsloth.py --config configs/training_gpu.yaml

    echo ""
    echo "=== Training complete ==="

    # Post-training FAQ inference on the same instance (uses warm GPU, the
    # merged model already on disk). Skip with SKIP_INFERENCE=true.
    if [ "${SKIP_INFERENCE:-false}" != "true" ]; then
        echo ""
        echo "=== Running post-training FAQ inference ==="
        python scripts/remote_inference.py --config configs/training_gpu.yaml || \
            echo "WARNING: inference failed, but training artifacts are intact"
        echo ""
        echo "=== Inference complete ==="
    fi

    echo "Download artifacts before shutdown:"
    echo "  scp -r ubuntu@<IP>:~/food-cpg-intelligence/models/adapters/ ."
    echo "  scp -r ubuntu@<IP>:~/food-cpg-intelligence/data/evaluation/faq_responses_*.jsonl ."

    if [ "${AUTO_SHUTDOWN:-true}" = "true" ]; then
        echo ""
        echo "!!! AUTO-SHUTDOWN in 15 minutes !!!"
        sudo shutdown -h +15 "Training + inference complete — auto-shutdown"
    fi
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# No flag — show usage
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "SKUFood Training"
echo ""
echo "Usage:"
echo "  bash scripts/remote_train.sh --docker            # Run training in Docker (recommended)"
echo "  bash scripts/remote_train.sh --docker --dry-run  # Validate config/data only"
echo "  bash scripts/remote_train.sh --package-docker    # Build + export Docker image"
echo "  bash scripts/remote_train.sh --bare              # Bare-metal (legacy fallback)"
echo "  bash scripts/remote_train.sh --package           # Package for bare-metal upload"
exit 1

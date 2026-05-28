# SKUFood QLoRA Training Container
# Pins ALL dependencies so training is reproducible and testable locally.
#
# Build:  docker build -t skufood-train -f docker/training/Dockerfile .
# Run:    docker run --gpus all -v ./data:/workspace/data -v ./models:/workspace/models skufood-train
# Test:   docker run --gpus all skufood-train --dry-run

FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (libssl for llama.cpp GGUF export)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git cmake build-essential \
    libssl-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /workspace

# ──── Pin dependencies in a requirements file baked into the image ────
COPY docker/training/requirements-train.txt /workspace/requirements-train.txt

# Install unsloth first — it pins torch, transformers, datasets to compatible versions.
# DO NOT pip install --upgrade transformers/datasets/torch separately.
RUN pip install --upgrade pip && \
    pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" && \
    pip install --upgrade --force-reinstall --no-cache-dir --no-deps unsloth_zoo && \
    pip install -r /workspace/requirements-train.txt && \
    pip install --no-build-isolation "flash-attn>=2.5.0" && \
    pip install --no-build-isolation causal-conv1d && \
    pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git" && \
    python -c "import unsloth; import transformers; print(f'transformers={transformers.__version__}')"

# Copy training script and config into image
COPY scripts/remote_train_unsloth.py /workspace/scripts/remote_train_unsloth.py
COPY configs/training_gpu.yaml /workspace/configs/training_gpu.yaml

# Validate the install (catches import errors at build time, not runtime)
RUN python -c "\
import unsloth; \
import transformers; \
import trl; \
import peft; \
print(f'unsloth OK'); \
print(f'transformers {transformers.__version__}'); \
print(f'trl {trl.__version__}'); \
print(f'peft {peft.__version__}'); \
"

ENTRYPOINT ["python", "scripts/remote_train_unsloth.py", "--config", "configs/training_gpu.yaml"]

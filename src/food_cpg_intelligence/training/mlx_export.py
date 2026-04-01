"""Model export pipeline: fuse LoRA adapter → GGUF → Ollama registration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

from food_cpg_intelligence.training.formatter import SYSTEM_PROMPT

logger = structlog.stdlib.get_logger(__name__)


def fuse_adapter(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
) -> Path:
    """Fuse LoRA adapter weights into the base model.

    Uses mlx_lm.fuse to merge adapter into a standalone model.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "mlx_lm.fuse",
        "--model",
        base_model,
        "--adapter-path",
        str(adapter_path),
        "--save-path",
        str(output_path),
    ]
    logger.info("fuse_start", cmd=" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"mlx_lm.fuse failed: {result.stderr[:500]}")

    logger.info("fuse_complete", output=str(output_path))
    return output_path


def convert_to_gguf(
    fused_model_path: Path,
    output_path: Path,
    *,
    quantization: str = "q4_k_m",
) -> Path:
    """Convert fused MLX model to GGUF format for Ollama.

    Tries mlx_lm.convert first, falls back to llama.cpp convert script.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try mlx_lm convert (outputs safetensors/GGUF depending on version)
    cmd = [
        "python",
        "-m",
        "mlx_lm.convert",
        "--model",
        str(fused_model_path),
        "--quantize",
        "--q-bits",
        "4",
        "-o",
        str(output_path.parent),
    ]
    logger.info("convert_start", method="mlx_lm.convert", cmd=" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        logger.info("convert_complete", output=str(output_path))
        return output_path

    logger.warning("mlx_convert_failed", stderr=result.stderr[:200], fallback="llama.cpp")

    # Fallback: try llama.cpp's convert script
    llama_convert = _find_llama_cpp_convert()
    if llama_convert is None:
        raise RuntimeError(
            "GGUF conversion failed. Install llama.cpp:\n"
            "  brew install llama.cpp\n"
            "  # or: git clone https://github.com/ggerganov/llama.cpp"
        )

    cmd = [
        "python",
        str(llama_convert),
        str(fused_model_path),
        "--outfile",
        str(output_path),
        "--outtype",
        quantization,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"llama.cpp convert failed: {result.stderr[:500]}")

    logger.info("convert_complete", method="llama.cpp", output=str(output_path))
    return output_path


def _find_llama_cpp_convert() -> Path | None:
    """Try to find llama.cpp's convert_hf_to_gguf.py."""
    candidates = [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/share/llama.cpp/convert_hf_to_gguf.py"),
        Path("/opt/homebrew/share/llama.cpp/convert_hf_to_gguf.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def generate_modelfile(
    model_path: Path,
    output_path: Path,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.3,
    context_length: int = 4096,
) -> Path:
    """Generate an Ollama Modelfile for the fine-tuned model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modelfile = f"""FROM {model_path}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature {temperature}
PARAMETER num_ctx {context_length}
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
"""
    output_path.write_text(modelfile, encoding="utf-8")
    logger.info("modelfile_generated", path=str(output_path))
    return output_path


def register_with_ollama(
    modelfile_path: Path,
    model_name: str,
) -> None:
    """Register a model with Ollama using the generated Modelfile."""
    cmd = ["ollama", "create", model_name, "-f", str(modelfile_path)]
    logger.info("ollama_create", cmd=" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ollama create failed: {result.stderr[:500]}")

    # Verify registration
    verify = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
    if model_name in verify.stdout:
        logger.info("ollama_registered", model=model_name)
    else:
        logger.warning("ollama_verify_failed", model=model_name, output=verify.stdout[:200])

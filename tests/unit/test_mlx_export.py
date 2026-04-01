"""Tests for MLX export pipeline."""

from pathlib import Path

from food_cpg_intelligence.training.mlx_export import generate_modelfile


def test_generate_modelfile(tmp_path: Path) -> None:
    model_path = tmp_path / "skufood.gguf"
    model_path.touch()  # create empty file

    modelfile = tmp_path / "Modelfile"
    result = generate_modelfile(model_path, modelfile)

    assert result == modelfile
    assert modelfile.exists()

    content = modelfile.read_text()
    assert f"FROM {model_path}" in content
    assert "SYSTEM" in content
    assert "temperature" in content
    assert "num_ctx" in content
    assert "<|im_end|>" in content  # stop token


def test_generate_modelfile_custom_params(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.touch()

    modelfile = tmp_path / "Modelfile"
    generate_modelfile(
        model_path,
        modelfile,
        system_prompt="Custom prompt",
        temperature=0.7,
        context_length=8192,
    )

    content = modelfile.read_text()
    assert "Custom prompt" in content
    assert "temperature 0.7" in content
    assert "num_ctx 8192" in content

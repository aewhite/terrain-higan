from __future__ import annotations

import copy
import json
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn as nn
from safetensors.torch import save_file

from terrain_higan.etopo.normalization import ELEVATION_MEAN_M, ELEVATION_STD_M
from terrain_higan.projects.z0_generator.config import (
    GENERATOR_METADATA_FILENAME,
    GENERATOR_ONNX_FILENAME,
    GENERATOR_PT2_FILENAME,
    GENERATOR_SHRINK,
    GENERATOR_STATE_DICT_FILENAME,
    Z0GeneratorConfig,
)
from terrain_higan.projects.z0_generator.models import Z0Generator


class Z0GeneratorExportModule(nn.Module):
    def __init__(self, generator: Z0Generator) -> None:
        super().__init__()
        self.generator = generator

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.generator(noise)


EXPORT_MINIMUM_OUTPUT_SIZE = 32
EXPORT_MINIMUM_INPUT_SIZE = GENERATOR_SHRINK + EXPORT_MINIMUM_OUTPUT_SIZE


def export_z0_generator(
        generator: Z0Generator,
        config: Z0GeneratorConfig,
        *,
        output_dir: Path | None = None,
        update_latest: bool = True,
) -> tuple[Path, Path, Path, Path]:
    if output_dir is None:
        output_dir = config.model_run_dir
    elif update_latest:
        raise ValueError("update_latest requires the configured model_run_dir")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict_path = output_dir / GENERATOR_STATE_DICT_FILENAME
    pt2_path = output_dir / GENERATOR_PT2_FILENAME
    onnx_path = output_dir / GENERATOR_ONNX_FILENAME
    metadata_path = output_dir / GENERATOR_METADATA_FILENAME
    temporary_state_dict_path = _temporary_path(state_dict_path)
    temporary_pt2_path = _temporary_path(pt2_path)
    temporary_onnx_path = _temporary_path(onnx_path)
    temporary_metadata_path = _temporary_path(metadata_path)

    try:
        export_generator = copy.deepcopy(generator).to("cpu").eval()
        exported_program, onnx_program = _build_exported_programs(export_generator, config)
        save_file(export_generator.state_dict(), temporary_state_dict_path)
        torch.export.save(exported_program, temporary_pt2_path)
        onnx_program.save(temporary_onnx_path)
        _write_metadata(temporary_metadata_path, config, onnx_program)

        temporary_state_dict_path.replace(state_dict_path)
        temporary_pt2_path.replace(pt2_path)
        temporary_onnx_path.replace(onnx_path)
        temporary_metadata_path.replace(metadata_path)
    finally:
        for temporary_path in (
                temporary_state_dict_path,
                temporary_pt2_path,
                temporary_onnx_path,
                temporary_metadata_path,
        ):
            _remove_if_exists(temporary_path)

    if update_latest:
        _update_latest_symlink(config.model_dir, output_dir)

    return state_dict_path, pt2_path, onnx_path, metadata_path


def _build_exported_programs(
        export_generator: Z0Generator,
        config: Z0GeneratorConfig,
) -> tuple[torch.export.ExportedProgram, torch.onnx.ONNXProgram]:
    export_module = Z0GeneratorExportModule(export_generator).eval()
    example_noise = torch.randn(2, 1, config.noise_size, config.noise_size, dtype=torch.float32)
    batch = torch.export.Dim("batch", min=1)
    height = torch.export.Dim("height", min=EXPORT_MINIMUM_INPUT_SIZE)
    width = torch.export.Dim("width", min=EXPORT_MINIMUM_INPUT_SIZE)
    dynamic_shapes = {"noise": {0: batch, 2: height, 3: width}}

    with torch.inference_mode():
        exported_program = torch.export.export(
            export_module,
            (example_noise,),
            dynamic_shapes=dynamic_shapes,
        )
        onnx_program = torch.onnx.export(
            exported_program,
            args=(),
            input_names=["noise"],
            output_names=["elevation_normalized"],
            dynamo=True,
            external_data=False,
        )
    if onnx_program is None:
        raise RuntimeError("ONNX export did not produce a program")

    return exported_program, onnx_program


def _write_metadata(
        path: Path,
        config: Z0GeneratorConfig,
        onnx_program: torch.onnx.ONNXProgram,
) -> None:
    opset_versions = {
        opset.domain or "ai.onnx": opset.version
        for opset in onnx_program.model_proto.opset_import
    }
    metadata = {
        "format_version": 2,
        "model_type": "z0_generator",
        "artifacts": {
            "state_dict": GENERATOR_STATE_DICT_FILENAME,
            "pt2": GENERATOR_PT2_FILENAME,
            "onnx": GENERATOR_ONNX_FILENAME,
        },
        "input": {
            "name": "noise",
            "dtype": "float32",
            "shape": ["batch", 1, "height", "width"],
            "distribution": "standard_normal",
            "minimum_height": EXPORT_MINIMUM_INPUT_SIZE,
            "minimum_width": EXPORT_MINIMUM_INPUT_SIZE,
        },
        "output": {
            "name": "elevation_normalized",
            "dtype": "float32",
            "shape": ["batch", 1, "height - generator_shrink", "width - generator_shrink"],
            "units": "normalized",
        },
        "elevation_normalization": {
            "mean_m": ELEVATION_MEAN_M,
            "std_m": ELEVATION_STD_M,
        },
        "generator_shrink": GENERATOR_SHRINK,
        "torch_version": torch.__version__,
        "onnx_opset_versions": opset_versions,
        "export_example_noise_shape": [2, 1, config.noise_size, config.noise_size],
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def _update_latest_symlink(model_dir: Path, model_run_dir: Path) -> None:
    latest_path = model_dir / "latest"
    temporary_latest_path = _temporary_path(latest_path)
    relative_target = model_run_dir.relative_to(model_dir)
    try:
        temporary_latest_path.symlink_to(relative_target, target_is_directory=True)
        temporary_latest_path.replace(latest_path)
    finally:
        _remove_if_exists(temporary_latest_path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid4().hex}.tmp")


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

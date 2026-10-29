from __future__ import annotations

import copy
from pathlib import Path
from uuid import uuid4

import torch
from safetensors.torch import load_file, save_file

from terrain_higan.projects.zn_map_upscaler.config import (
    MODEL_FILENAME,
    ONNX_FILENAME,
    PT2_FILENAME,
    Z_LEVEL_CONFIGS,
    ZnUpscalerConfig,
)
from terrain_higan.models.upscaling import upscaler_convolution_weights
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel


def save_zn_map_upscaler(
        model: ZnUpscalerModel,
        config: ZnUpscalerConfig,
        *,
        output_dir: Path | None = None,
        update_latest: bool = True,
) -> tuple[Path, Path, Path]:
    if output_dir is None:
        output_dir = config.model_run_dir
    elif update_latest:
        raise ValueError("update_latest requires the configured model_run_dir")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    pt2_path = output_dir / PT2_FILENAME
    onnx_path = output_dir / ONNX_FILENAME
    temporary_model_path = _temporary_path(model_path)
    temporary_pt2_path = _temporary_path(pt2_path)
    temporary_onnx_path = _temporary_path(onnx_path)
    try:
        export_model = copy.deepcopy(model).to("cpu").eval()
        example_input_map = torch.randn(2, 1, 32, 32, dtype=torch.float32)
        example_noise = torch.randn(2, 4, 32, 32, dtype=torch.float32)
        batch = torch.export.Dim("batch", min=1)
        height = torch.export.Dim("height", min=9)
        width = torch.export.Dim("width", min=9)
        dynamic_shapes = {
            "input_map": {0: batch, 2: height, 3: width},
            "noise": {0: batch, 2: height, 3: width},
        }
        with torch.inference_mode():
            exported_program = torch.export.export(
                export_model,
                (example_input_map, example_noise),
                dynamic_shapes=dynamic_shapes,
            )
            onnx_program = torch.onnx.export(
                exported_program,
                args=(),
                input_names=["input_map", "noise"],
                output_names=["elevation_normalized"],
                dynamo=True,
                external_data=False,
            )
        if onnx_program is None:
            raise RuntimeError("ONNX export did not produce a program")
        save_file(export_model.state_dict(), temporary_model_path)
        torch.export.save(exported_program, temporary_pt2_path)
        onnx_program.save(temporary_onnx_path)
        temporary_model_path.replace(model_path)
        temporary_pt2_path.replace(pt2_path)
        temporary_onnx_path.replace(onnx_path)
    finally:
        for path in (temporary_model_path, temporary_pt2_path, temporary_onnx_path):
            _remove_if_exists(path)

    if update_latest:
        _update_latest_symlink(config.model_dir / config.level_name, output_dir)
    return model_path, pt2_path, onnx_path


def load_zn_map_upscaler(
        model_path: Path,
        *,
        z_level: int | None = None,
        layer_channels: tuple[int, ...] | None = None,
) -> ZnUpscalerModel:
    resolved_model_path = _resolve_model_path(model_path)
    if resolved_model_path.suffix == ".safetensors":
        state_dict = load_file(resolved_model_path, device="cpu")
    else:
        state_dict = torch.load(resolved_model_path, map_location="cpu", weights_only=True)
    if z_level is None:
        z_level = _stats_z_level_from_model_path(model_path)
    if layer_channels is None:
        layer_channels = _layer_channels_from_state_dict(state_dict)
    model = ZnUpscalerModel(z_level=z_level, layer_channels=layer_channels)
    model.load_state_dict(state_dict)
    return model.eval()


def _resolve_model_path(model_path: Path) -> Path:
    if model_path.is_file() or model_path.suffix != ".safetensors":
        return model_path

    legacy_model_path = model_path.with_suffix(".pt")
    if legacy_model_path.is_file():
        return legacy_model_path
    return model_path


def _stats_z_level_from_model_path(model_path: Path) -> int:
    label = model_path.parent.parent.name
    for level in Z_LEVEL_CONFIGS:
        if level.level_name == label:
            return level.z_level
    raise ValueError(
        f"Cannot infer stats z level from {model_path}; pass z_level when loading this artifact"
    )


def _layer_channels_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    return tuple(weight.shape[0] for weight in upscaler_convolution_weights(state_dict))


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
    return path.with_name(f".{path.stem}.{uuid4().hex}{path.suffix}")


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

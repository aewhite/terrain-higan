"""Loading compatible project safetensors checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from terrain_higan.models.upscaling import upscaler_convolution_weights

from .models import Upscaler, Z0Generator


def checkpoint_path(zoom: int) -> Path:
    if zoom == 0:
        return Path("z0_generator/latest/generator.safetensors")
    name = f"z{zoom}s" if zoom > 10 else f"z{zoom}"
    return Path("zn_map_upscaler") / name / "latest/upscaler.safetensors"


def default_weights_dir() -> Path:
    source_root = Path(__file__).resolve().parents[3]
    development = (source_root / "src/terrain_higan/paths.py").is_file()
    packaged = Path(__file__).with_name("weights")
    if packaged.is_dir():
        _verify_packaged_weights(packaged)
        return packaged
    if development:
        return source_root / ".local" / "weights"
    raise FileNotFoundError("No packaged weights; pass weights_dir or use --weights-dir")


def _verify_packaged_weights(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["format_version"] != 1 or manifest["architecture"] != "wfcai-map-inference-v1":
        raise ValueError("Unsupported packaged weights manifest")
    expected_paths = {checkpoint_path(zoom).as_posix() for zoom in range(21)}
    records = manifest["weights"]
    if len(records) != 21 or {record["path"] for record in records} != expected_paths:
        raise ValueError("Packaged weights manifest must contain exactly z0 through z10 plus z11s through z20s")
    for record in records:
        path = directory / record["path"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Packaged weight checksum mismatch: {record['path']}")


def load_z0(path: Path) -> Z0Generator:
    model = Z0Generator()
    _load(model, path)
    return model.eval()


def load_upscaler(path: Path, *, checkpoint_zoom: int) -> Upscaler:
    state_dict = _state_dict(path)
    layer_channels = _layer_channels(state_dict)
    model = Upscaler(checkpoint_zoom=checkpoint_zoom, layer_channels=layer_channels)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Incompatible upscaler checkpoint {path}: {error}") from error
    return model.eval()


def _load(model: torch.nn.Module, path: Path) -> None:
    try:
        model.load_state_dict(_state_dict(path), strict=True)
    except RuntimeError as error:
        raise ValueError(f"Incompatible checkpoint {path}: {error}") from error


def _state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state_dict = load_file(path, device="cpu")
    if not state_dict:
        raise ValueError(f"Checkpoint {path} is empty")
    for name, value in state_dict.items():
        if value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint {path} has invalid tensor {name}")
    return state_dict


def _layer_channels(state_dict: dict[str, torch.Tensor]) -> tuple[int, ...]:
    return tuple(int(weight.shape[0]) for weight in upscaler_convolution_weights(state_dict))

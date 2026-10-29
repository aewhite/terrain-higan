"""Generate one large terrain map through the trained z(n) upscaler chain."""

from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

import matplotlib.image as mpl_image
import numpy as np
import torch
from tqdm import tqdm

from terrain_higan.etopo.normalization import denormalize_elevation
from terrain_higan.paths import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR
from terrain_higan.projects.z0_generator.config import GENERATOR_SHRINK
from terrain_higan.projects.zn_map_upscaler.config import MODEL_FILENAME
from terrain_higan.projects.zn_map_upscaler.save import load_zn_map_upscaler
from terrain_higan.visualization.terrain import (
    DEFAULT_ARRAY_MAX_ELEV_M,
    DEFAULT_MIN_ELEV_M,
    build_tensorboard_terrain_colormap,
)


DEFAULT_Z0_SIZE = 64
DEFAULT_Z_LEVEL = 7
DEFAULT_TILE_INPUT_SIZE = 256
UPSCALER_SHRINK = 16
UPSCALER_INPUT_OVERLAP = UPSCALER_SHRINK // 2


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z0-size", type=int, default=DEFAULT_Z0_SIZE)
    parser.add_argument("--z-level", type=int, choices=range(1, 11), default=DEFAULT_Z_LEVEL)
    parser.add_argument(
        "--seed",
        type=int,
        help="Generation seed; defaults to a new random seed for each run",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-input-size", type=int, default=DEFAULT_TILE_INPUT_SIZE)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR / "zn_map_upscaler",
    )
    parser.add_argument(
        "--z0-model-path",
        type=Path,
        default=DEFAULT_MODELS_DIR / "z0_generator" / "latest" / "generator.pt2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR / "generated_maps",
    )
    return parser.parse_args()


def calculate_level_sizes(z0_size: int, z_level: int) -> tuple[int, ...]:
    """Return z0..zn output sizes for a square z0 noise input."""
    if z0_size <= GENERATOR_SHRINK:
        raise ValueError(
            f"z0_size must be greater than the z0 shrink of {GENERATOR_SHRINK}, got {z0_size}"
        )
    if not 1 <= z_level <= 10:
        raise ValueError(f"z_level must be in 1..10, got {z_level}")

    sizes = [z0_size - GENERATOR_SHRINK]
    for _ in range(z_level):
        sizes.append(2 * sizes[-1] - UPSCALER_SHRINK)
    return tuple(sizes)


def generate_large_map(
    *,
    z0_size: int,
    z_level: int,
    seed: int,
    device: torch.device,
    tile_input_size: int,
    model_dir: Path,
    z0_model_path: Path,
) -> torch.Tensor:
    level_sizes = calculate_level_sizes(z0_size, z_level)
    if tile_input_size <= UPSCALER_INPUT_OVERLAP:
        raise ValueError(
            f"tile_input_size must be greater than {UPSCALER_INPUT_OVERLAP}, "
            f"got {tile_input_size}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    random = torch.Generator(device="cpu")
    random.manual_seed(seed)
    z0_model = torch.export.load(z0_model_path).module().to(device)
    z0_noise = torch.randn(1, 1, z0_size, z0_size, generator=random)
    with torch.inference_mode():
        current_map = z0_model(z0_noise.to(device)).to("cpu", dtype=torch.float32)
    del z0_model, z0_noise
    _validate_map_size(current_map, level_sizes[0], "z0")

    for level in range(1, z_level + 1):
        model_path = model_dir / f"z{level}" / "latest" / MODEL_FILENAME
        model = load_zn_map_upscaler(model_path).to(device).eval()
        noise = torch.randn(
            1,
            4,
            current_map.shape[-2],
            current_map.shape[-1],
            generator=random,
            dtype=current_map.dtype,
        )
        current_map = _upscale_tiled(
            model,
            current_map,
            noise,
            device=device,
            tile_input_size=tile_input_size,
            description=f"Upscaling z{level - 1} -> z{level}",
        )
        _validate_map_size(current_map, level_sizes[level], f"z{level}")
        del model, noise
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return current_map


def _upscale_tiled(
    model: torch.nn.Module,
    input_map: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    tile_input_size: int,
    description: str,
) -> torch.Tensor:
    """Run an upscaler in overlapping input tiles and stitch its exact output."""
    height, width = input_map.shape[-2:]
    output = torch.empty(
        1,
        1,
        2 * height - UPSCALER_SHRINK,
        2 * width - UPSCALER_SHRINK,
        dtype=torch.float32,
        device="cpu",
    )
    row_starts = _tile_starts(height, tile_input_size)
    column_starts = _tile_starts(width, tile_input_size)

    with torch.inference_mode(), tqdm(
        total=len(row_starts) * len(column_starts),
        desc=description,
        unit="tile",
    ) as progress:
        for row_start in row_starts:
            row_end = min(row_start + tile_input_size, height)
            output_row_start = 2 * row_start
            for column_start in column_starts:
                column_end = min(column_start + tile_input_size, width)
                output_column_start = 2 * column_start
                tile = model(
                    input_map[..., row_start:row_end, column_start:column_end].to(device),
                    noise[..., row_start:row_end, column_start:column_end].to(device),
                ).to("cpu", dtype=torch.float32)
                tile_height, tile_width = tile.shape[-2:]
                output[
                    ...,
                    output_row_start:output_row_start + tile_height,
                    output_column_start:output_column_start + tile_width,
                ] = tile
                progress.update()
    return output


def _tile_starts(size: int, tile_input_size: int) -> tuple[int, ...]:
    if size <= tile_input_size:
        return (0,)
    stride = tile_input_size - UPSCALER_INPUT_OVERLAP
    starts = [0]
    while starts[-1] + tile_input_size < size:
        starts.append(starts[-1] + stride)
    return tuple(starts)


def _validate_map_size(generated_map: torch.Tensor, expected_size: int, level: str) -> None:
    expected_shape = (1, 1, expected_size, expected_size)
    if tuple(generated_map.shape) != expected_shape:
        raise ValueError(
            f"Expected {level} map shape {expected_shape}, got {tuple(generated_map.shape)}"
        )


def save_large_map(
    terrain: torch.Tensor,
    *,
    output_dir: Path,
    z0_size: int,
    z_level: int,
    seed: int,
    model_dir: Path,
    z0_model_path: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    size = terrain.shape[-1]
    basename = f"terrain_z{z_level}_seed{seed}_{size}x{size}"
    normalized_path = output_dir / f"{basename}.normalized.npy"
    preview_path = output_dir / f"{basename}.png"
    metadata_path = output_dir / f"{basename}.json"

    normalized = terrain[0, 0].numpy()
    np.save(normalized_path, normalized)
    meters = np.asarray(denormalize_elevation(normalized), dtype=np.float32)
    mpl_image.imsave(
        preview_path,
        meters,
        cmap=build_tensorboard_terrain_colormap(
            min_elev=DEFAULT_MIN_ELEV_M,
            max_elev=DEFAULT_ARRAY_MAX_ELEV_M,
        ),
        vmin=DEFAULT_MIN_ELEV_M,
        vmax=DEFAULT_ARRAY_MAX_ELEV_M,
        origin="upper",
    )
    metadata = {
        "z0_noise_size": z0_size,
        "z_level": z_level,
        "level_output_sizes": calculate_level_sizes(z0_size, z_level),
        "seed": seed,
        "shape": list(normalized.shape),
        "data_file": normalized_path.name,
        "data_units": "universal_normalized_elevation",
        "preview_file": preview_path.name,
        "z0_model_path": str(z0_model_path),
        "upscaler_model_dir": str(model_dir),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return normalized_path, preview_path, metadata_path


def main() -> None:
    args = parse_arguments()
    device = torch.device(args.device)
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    level_sizes = calculate_level_sizes(args.z0_size, args.z_level)
    print(
        f"Generating z0..z{args.z_level} terrain from {args.z0_size}x{args.z0_size} "
        f"z0 noise with seed {seed}; final size is {level_sizes[-1]}x{level_sizes[-1]}"
    )
    terrain = generate_large_map(
        z0_size=args.z0_size,
        z_level=args.z_level,
        seed=seed,
        device=device,
        tile_input_size=args.tile_input_size,
        model_dir=args.model_dir,
        z0_model_path=args.z0_model_path,
    )
    paths = save_large_map(
        terrain,
        output_dir=args.output_dir,
        z0_size=args.z0_size,
        z_level=args.z_level,
        seed=seed,
        model_dir=args.model_dir,
        z0_model_path=args.z0_model_path,
    )
    print("Saved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()

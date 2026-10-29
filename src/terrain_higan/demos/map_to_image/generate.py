"""Generate an exact-size terrain region and render it with local colormaps."""

from __future__ import annotations

import argparse
import json
import time
from importlib.util import find_spec
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import matplotlib.image as mpl_image
import numpy as np
import torch

from terrain_higan.inference import InferenceConfig, Region, TerrainWorld, denormalize_elevation, meters_per_pixel

from .colormaps import FROST_LINE_M, render

DEFAULT_COLORMAPS = ("greyscale", "elevation", "realistic")
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def default_output_dir() -> Path:
    if find_spec("terrain_higan.paths") is not None:
        from terrain_higan.paths import DEFAULT_RUNS_DIR

        return DEFAULT_RUNS_DIR / "map_to_image"
    return Path("runs/map_to_image")


class TileWorld(Protocol):
    config: InferenceConfig

    def iter_tiles(self, region: Region) -> Iterator[tuple[int, int, torch.Tensor]]: ...


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--z-level", type=int, default=7)
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--cache-mib", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--weights-dir", type=Path, help="Checkpoint directory; defaults to bundled or checkout weights")
    parser.add_argument("--colormaps", nargs="+", choices=DEFAULT_COLORMAPS, default=list(DEFAULT_COLORMAPS))
    return parser.parse_args(argv)


def resolve_device(device_override: str | None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_seed() -> int:
    return time.time_ns() // 1_000_000


def encode_seed(seed: int) -> str:
    if not 0 <= seed < 2 ** 64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if seed == 0:
        return BASE62_ALPHABET[0]
    encoded: list[str] = []
    while seed:
        seed, remainder = divmod(seed, len(BASE62_ALPHABET))
        encoded.append(BASE62_ALPHABET[remainder])
    return "".join(reversed(encoded))


def generate_normalized_map(world: TileWorld, region: Region, output_path: Path) -> np.memmap:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    terrain = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(region.height, region.width))
    tile_size = world.config.tile_size
    for tile_x, tile_y, tile in world.iter_tiles(region):
        tile_left = tile_x * tile_size
        tile_top = tile_y * tile_size
        left = max(region.x, tile_left)
        top = max(region.y, tile_top)
        right = min(region.x + region.width, tile_left + tile_size)
        bottom = min(region.y + region.height, tile_top + tile_size)
        tile_array = tile[0, 0].detach().cpu().numpy()
        terrain[top - region.y:bottom - region.y, left - region.x:right - region.x] = tile_array[
            top - tile_top:bottom - tile_top, left - tile_left:right - tile_left
        ]
    terrain.flush()
    return terrain


def output_basename(region: Region, seed: int) -> str:
    return f"terrain_z{region.zoom}_seed{encode_seed(seed)}_x{region.x}_y{region.y}_{region.width}x{region.height}"


def save_renderings(normalized: np.ndarray, *, output_dir: Path, basename: str, colormaps: list[str]) -> dict[str, Path]:
    meters = np.asarray(denormalize_elevation(normalized), dtype=np.float32)
    paths: dict[str, Path] = {}
    for colormap in colormaps:
        path = output_dir / f"{basename}.{colormap}.png"
        mpl_image.imsave(path, render(colormap, meters))
        paths[colormap] = path
    return paths


def build_metadata(
    region: Region,
    seed: int,
    config: InferenceConfig,
    *,
    data_file: Path,
    images: dict[str, Path],
) -> dict[str, object]:
    return {
        "seed": seed,
        "encoded_seed": encode_seed(seed),
        "zoom": region.zoom,
        "x": region.x,
        "y": region.y,
        "shape": [region.height, region.width],
        "meters_per_pixel": meters_per_pixel(region.zoom),
        "device": str(config.device),
        "tile_size": config.tile_size,
        "cache_mib": config.cache_bytes // (1024 * 1024),
        "data_file": data_file.name,
        "data_units": "universal_normalized_elevation",
        "images": {name: path.name for name, path in images.items()},
        "realistic_frost_line_m": FROST_LINE_M,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    region = Region(args.z_level, args.x, args.y, args.width, args.height)
    seed = default_seed() if args.seed is None else args.seed
    config = InferenceConfig(
        device=resolve_device(args.device),
        tile_size=args.tile_size,
        cache_bytes=args.cache_mib * 1024 * 1024,
    )
    # Naming validates the seed before expensive model loading or file creation.
    basename = output_basename(region, seed)
    world = TerrainWorld.from_default_weights(seed=seed, config=config, weights_dir=args.weights_dir)
    normalized_path = args.output_dir / f"{basename}.normalized.npy"
    normalized = generate_normalized_map(world, region, normalized_path)
    render_paths = save_renderings(
        normalized,
        output_dir=args.output_dir,
        basename=basename,
        colormaps=args.colormaps,
    )
    metadata = build_metadata(region, seed, config, data_file=normalized_path, images=render_paths)
    metadata_path = args.output_dir / f"{basename}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Seed: {seed} ({encode_seed(seed)})")
    print(f"Saved normalized terrain: {normalized_path}")
    for path in render_paths.values():
        print(f"Saved image: {path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()

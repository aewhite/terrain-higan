"""Generate fixed-size map batches for z(n) upscaler training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from terrain_higan.paths import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR
from terrain_higan.models.upscaling import UPSCALER_SHRINK
from terrain_higan.projects.z0_generator.config import GENERATOR_SHRINK
from terrain_higan.projects.zn_map_upscaler.config import MODEL_FILENAME, Z_LEVEL_CONFIGS
from terrain_higan.projects.zn_map_upscaler.geometry import center_crop
from terrain_higan.projects.zn_map_upscaler.save import load_zn_map_upscaler

DEFAULT_BATCH_SIZE = 512
DEFAULT_FILE_BATCH_SIZE = 8_192
DEFAULT_TOTAL_SAMPLES = 1_048_576
DEFAULT_MAP_SIZE = 32
DEFAULT_MAPS_DIR = DEFAULT_DATA_DIR / "maps"
Z_LEVELS = range(0, 21)


class MapBatchSource:
    """Sequentially read B1HW samples from generated NumPy batch files."""

    def __init__(self, directory: Path, *, map_size: int, start_sample: int = 0) -> None:
        self._paths = sorted(directory.glob("maps_*.npy"))
        if not self._paths:
            raise FileNotFoundError(f"No generated map batches found in {directory}")

        self._batches = [np.load(path, mmap_mode="r") for path in self._paths]
        for path, batch in zip(self._paths, self._batches):
            if batch.ndim != 4 or batch.shape[1:] != (1, map_size, map_size):
                raise ValueError(
                    f"Expected B1{map_size}x{map_size} maps in {path}, got {batch.shape}"
                )
        self._sample_count = sum(batch.shape[0] for batch in self._batches)
        if self._sample_count == 0:
            raise ValueError(f"Generated map batches in {directory} contain no samples")
        self._batch_index = 0
        self._offset = 0
        self._advance(start_sample % self._sample_count)

    def take(self, count: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        remaining = count
        while remaining:
            batch = self._batches[self._batch_index]
            take_count = min(remaining, batch.shape[0] - self._offset)
            parts.append(batch[self._offset:self._offset + take_count])
            self._advance(take_count)
            remaining -= take_count
        return np.concatenate(parts, axis=0)

    def _advance(self, count: int) -> None:
        while count:
            batch = self._batches[self._batch_index]
            available = batch.shape[0] - self._offset
            step = min(count, available)
            self._offset += step
            count -= step
            if self._offset == batch.shape[0]:
                self._batch_index = (self._batch_index + 1) % len(self._batches)
                self._offset = 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z-level", type=int, choices=Z_LEVELS, required=True)
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--file-batch-size", type=int, default=DEFAULT_FILE_BATCH_SIZE)
    parser.add_argument("--map-size", type=int, default=DEFAULT_MAP_SIZE)
    parser.add_argument("--mode", choices=("rebuild", "append"), default="rebuild")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def generate_maps(
        *,
        z_level: int,
        level_name: str,
        total_samples: int,
        batch_size: int,
        file_batch_size: int,
        map_size: int,
        mode: str,
        output_dir: Path,
        source_dir: Path | None,
        model_dir: Path,
        z0_model_path: Path,
        device: torch.device,
        progress: bool,
) -> int:
    _validate_generation_request(
        z_level,
        level_name,
        total_samples,
        batch_size,
        file_batch_size,
        map_size,
        mode,
        device,
    )
    existing_samples = _prepare_output_dir(output_dir, mode, map_size)
    if existing_samples >= total_samples:
        return existing_samples

    if z_level == 0:
        model = torch.export.load(z0_model_path).module().to(device)
        source = None
    else:
        if source_dir is None:
            raise ValueError("source_dir is required for upscaler map generation")
        model_path = model_dir / level_name / "latest" / MODEL_FILENAME
        model = load_zn_map_upscaler(model_path).to(device).eval()
        source = MapBatchSource(source_dir, map_size=map_size, start_sample=existing_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    remaining_samples = total_samples - existing_samples
    batch_number = _next_batch_number(output_dir)
    input_size = (map_size + UPSCALER_SHRINK + 1) // 2
    # Match the source parity so centered input/output crops retain their pixel phase.
    input_size += (map_size - input_size) % 2
    with torch.inference_mode():
        progress_bar = _make_progress_bar(level_name, remaining_samples, progress)
        generated_samples = 0
        while generated_samples < remaining_samples:
            samples_in_file = min(file_batch_size, remaining_samples - generated_samples)
            file_batches: list[np.ndarray] = []
            samples_in_batch = 0
            while samples_in_batch < samples_in_file:
                current_batch_size = min(batch_size, samples_in_file - samples_in_batch)
                if z_level == 0:
                    maps = _generate_z0_maps(model, current_batch_size, map_size, device)
                else:
                    input_maps = torch.from_numpy(source.take(current_batch_size)).to(device)
                    noise = torch.randn(
                        current_batch_size,
                        4,
                        map_size,
                        map_size,
                        device=device,
                        dtype=input_maps.dtype,
                    )
                    maps = center_crop(model(
                        center_crop(input_maps, input_size),
                        center_crop(noise, input_size),
                    ), map_size)
                file_batches.append(maps.detach().to("cpu", dtype=torch.float32).numpy())
                samples_in_batch += current_batch_size
                progress_bar.update(current_batch_size)
            _save_batch(
                output_dir / f"maps_{batch_number:06d}.npy",
                np.concatenate(file_batches, axis=0),
            )
            batch_number += 1
            generated_samples += samples_in_file
        progress_bar.close()

    return total_samples


def _validate_generation_request(
        z_level: int,
        level_name: str,
        total_samples: int,
        batch_size: int,
        file_batch_size: int,
        map_size: int,
        mode: str,
        device: torch.device,
) -> None:
    if z_level not in Z_LEVELS:
        raise ValueError(f"z_level must be in 0..20, got {z_level}")
    expected_name = "z0" if z_level == 0 else Z_LEVEL_CONFIGS[z_level - 1].level_name
    if level_name != expected_name:
        raise ValueError(f"Expected level_name {expected_name!r} for z{z_level}, got {level_name!r}")
    if total_samples <= 0 or batch_size <= 0 or file_batch_size <= 0 or map_size <= 0:
        raise ValueError(
            "total_samples, batch_size, file_batch_size, and map_size must be positive"
        )
    if mode not in {"rebuild", "append"}:
        raise ValueError(f"Unsupported generation mode: {mode}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for map generation but is not available")


def _prepare_output_dir(output_dir: Path, mode: str, map_size: int) -> int:
    if mode == "rebuild" and output_dir.exists():
        shutil.rmtree(output_dir)
    if mode == "rebuild":
        return 0

    existing_samples = 0
    for path in sorted(output_dir.glob("maps_*.npy")):
        batch = np.load(path, mmap_mode="r")
        if batch.ndim != 4 or batch.shape[1:] != (1, map_size, map_size):
            raise ValueError(f"Expected B1{map_size}x{map_size} maps in {path}, got {batch.shape}")
        existing_samples += batch.shape[0]
    return existing_samples


def _generate_z0_maps(
        model: torch.nn.Module,
        batch_size: int,
        map_size: int,
        device: torch.device,
) -> torch.Tensor:
    noise_size = map_size + GENERATOR_SHRINK
    noise = torch.randn(batch_size, 1, noise_size, noise_size, device=device)
    return center_crop(model(noise), map_size)


def _next_batch_number(output_dir: Path) -> int:
    batch_numbers = [int(path.stem.removeprefix("maps_")) for path in output_dir.glob("maps_*.npy")]
    return max(batch_numbers, default=-1) + 1


def _make_progress_bar(level_label: str, total_samples: int, enabled: bool) -> tqdm:
    return tqdm(
        total=total_samples,
        desc=f"Generating {level_label} maps",
        unit="map",
        disable=not enabled,
    )


def _save_batch(path: Path, maps: np.ndarray) -> None:
    temporary_path = path.with_suffix(".tmp.npy")
    np.save(temporary_path, maps)
    temporary_path.replace(path)


def main() -> None:
    args = parse_arguments()
    current_name = "z0" if args.z_level == 0 else Z_LEVEL_CONFIGS[args.z_level - 1].level_name
    output_dir = args.output_dir or args.maps_dir / current_name
    source_dir = (
        args.maps_dir / ("z0" if args.z_level == 1 else Z_LEVEL_CONFIGS[args.z_level - 2].level_name)
        if args.z_level else None
    )
    sample_count = generate_maps(
        z_level=args.z_level,
        level_name=current_name,
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        file_batch_size=args.file_batch_size,
        map_size=args.map_size,
        mode=args.mode,
        output_dir=output_dir,
        source_dir=source_dir,
        model_dir=args.model_dir,
        z0_model_path=args.z0_model_path,
        device=torch.device(args.device),
        progress=args.progress,
    )
    print(f"Generated {sample_count:,} {current_name} maps in {output_dir}")


if __name__ == "__main__":
    main()

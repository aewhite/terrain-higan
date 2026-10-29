"""Analyze generated z(n-1) maps to build z(n) patch-selection PDFs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from terrain_higan.etopo.analyze_patches import (
    analyze_patch_means,
    save_patch_mean_chart,
    save_patch_mean_pdf,
)
from terrain_higan.models.zn_training_constants import PATCH_SIZES
from terrain_higan.paths import DEFAULT_DATA_DIR
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig

DEFAULT_PATCHES_PER_MAP = 4
DEFAULT_MAPS_DIR = DEFAULT_DATA_DIR / "maps"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "analysis"


@dataclass(frozen=True)
class InputMapPatchMeanAnalysis:
    z_level: int
    source_map_count: int
    patch_size: int
    patches_per_map: int
    pdf_path: Path
    histogram_path: Path

    @property
    def sample_count(self) -> int:
        return self.source_map_count * self.patches_per_map


def analyze_input_maps(
    *,
    z_level: int,
    maps_dir: Path = DEFAULT_MAPS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    source_level_name: str | None = None,
    output_level_name: str | None = None,
    patch_size: int | None = None,
    patches_per_map: int = DEFAULT_PATCHES_PER_MAP,
    seed: int | None = None,
    device: torch.device | str = "cuda",
) -> InputMapPatchMeanAnalysis:
    if z_level < 1 or (patch_size is None and z_level >= len(PATCH_SIZES)):
        raise ValueError(f"No patch size configured for z_level {z_level}")
    if patches_per_map <= 0:
        raise ValueError(f"patches_per_map must be positive, got {patches_per_map}")

    source_dir = maps_dir / (source_level_name or f"z{z_level - 1}")
    map_paths = sorted(source_dir.glob("maps_*.npy"))
    if not map_paths:
        raise FileNotFoundError(f"No materialized map batches found in {source_dir}")

    patch_size = patch_size if patch_size is not None else PATCH_SIZES[z_level] // 2 + 1
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    patch_means, source_map_count = _sample_patch_means(
        map_paths,
        patch_size=patch_size,
        patches_per_map=patches_per_map,
        random_generator=np.random.default_rng(seed),
    )
    patch_means_tensor = torch.from_numpy(patch_means).to(device)
    with torch.inference_mode():
        analysis = analyze_patch_means(patch_means_tensor)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_label = output_level_name or f"z{z_level}"
    pdf_path = output_dir / f"{output_label}_input_map_patch_mean_pdf.pt"
    histogram_path = output_dir / f"{output_label}_input_map_patch_mean_histogram.png"
    save_patch_mean_pdf(analysis.pdf, pdf_path)
    save_patch_mean_chart(
        analysis, histogram_path, title=f"{output_label} generated input-map patch means",
    )

    return InputMapPatchMeanAnalysis(
        z_level=z_level,
        source_map_count=source_map_count,
        patch_size=patch_size,
        patches_per_map=patches_per_map,
        pdf_path=pdf_path,
        histogram_path=histogram_path,
    )


def _sample_patch_means(
    map_paths: list[Path],
    *,
    patch_size: int,
    patches_per_map: int,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, int]:
    mean_batches: list[np.ndarray] = []
    source_map_count = 0
    patch_offsets = np.arange(patch_size)
    for path in map_paths:
        maps = np.load(path, mmap_mode="r")
        if maps.ndim != 4 or maps.shape[1] != 1:
            raise ValueError(f"Expected B1HW maps in {path}, got {maps.shape}")
        height, width = maps.shape[-2:]
        if height < patch_size or width < patch_size:
            raise ValueError(
                f"Cannot sample {patch_size}x{patch_size} patches from {path} with shape {maps.shape}"
            )
        map_count = maps.shape[0]
        rows = random_generator.integers(
            height - patch_size + 1,
            size=(map_count, patches_per_map),
        )
        columns = random_generator.integers(
            width - patch_size + 1,
            size=(map_count, patches_per_map),
        )
        patches = maps[
            np.arange(map_count)[:, None, None, None],
            0,
            rows[:, :, None, None] + patch_offsets[None, None, :, None],
            columns[:, :, None, None] + patch_offsets[None, None, None, :],
        ]
        mean_batches.append(np.asarray(patches.mean(axis=(-2, -1), dtype=np.float32)).reshape(-1))
        source_map_count += map_count
    return np.concatenate(mean_batches), source_map_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z-level", type=int, required=True)
    parser.add_argument("--real-dataset-z-level", type=int)
    parser.add_argument("--maps-dir", type=Path, default=DEFAULT_MAPS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--patches-per-map", type=int, default=DEFAULT_PATCHES_PER_MAP)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = ZnUpscalerConfig(
        z_level=args.z_level,
        real_dataset_z_level=args.real_dataset_z_level,
    )
    analysis = analyze_input_maps(
        z_level=args.z_level,
        maps_dir=args.maps_dir,
        output_dir=args.output_dir,
        source_level_name=config.predecessor_level_name,
        output_level_name=config.level_name,
        patch_size=config.input_metric_patch_size,
        patches_per_map=args.patches_per_map,
        seed=args.seed,
        device=args.device,
    )
    print(
        f"Analyzed {analysis.source_map_count:,} {config.predecessor_level_name} maps into "
        f"{analysis.sample_count:,} {analysis.patch_size}x{analysis.patch_size} patch means."
    )
    print(f"Saved selection PDF: {analysis.pdf_path}")
    print(f"Saved histogram: {analysis.histogram_path}")


if __name__ == "__main__":
    main()

"""Scan ETOPO patches, calibrate candidate constants, and write charts and reports."""
from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from terrain_higan.etopo.dataloader import DEFAULT_ETOPO_BASE_DIR, _discover_shards
from terrain_higan.etopo.normalization import (
    ELEVATION_MEAN_M,
    ELEVATION_STD_M,
    SYNTHETIC_SOURCE_Z_LEVEL,
    SYNTHETIC_STD_SCALES,
    UPSCALE_STATS,
    denormalize_elevation,
    normalize_elevation,
    scale_patch_deviations,
)
from terrain_higan.models.zn_training_constants import PATCH_SIZES
from terrain_higan.paths import DEFAULT_DATA_DIR

DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "analysis"
DEFAULT_SAMPLE_SIZE = 2_000_000
DEFAULT_BATCH_SIZE = 16_384
DEFAULT_SAMPLE_COUNT = 64_000
DEFAULT_SEED = 20260719
DEFAULT_GRID_SIZE = 4096
HISTOGRAM_CHUNK_VALUES = 4_194_304
DIVISOR_SEARCH_STEPS = 24
TARGET_TRANSFORMED_STDDEV = 1.0
ELEVATION_BINS = np.linspace(-8, 8, 257)
PATCH_STD_BINS = np.geomspace(1e-8, 1e3, 141)
DOWNSAMPLED_PATCH_STD_BINS = np.concatenate(([0.0], PATCH_STD_BINS))
STD_RATIO_BINS = np.linspace(0, 1, 101)
RESIDUAL_BINS = np.concatenate(
    (-np.geomspace(1e3, 1e-8, 140), [0.0], np.geomspace(1e-8, 1e3, 140))
)


@dataclass
class Moments:
    count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    m2: float = 0.0

    @property
    def stddev(self) -> float | None:
        return math.sqrt(max(0.0, self.m2 / self.count)) if self.count else None

    def update(self, values: torch.Tensor) -> None:
        if not values.numel():
            return
        variance, mean = torch.var_mean(values.double(), correction=0)
        minimum, maximum = torch.aminmax(values)
        self.merge(Moments(
            values.numel(), minimum.item(), maximum.item(), mean.item(),
            variance.item() * values.numel(),
        ))

    def merge(self, other: Moments) -> None:
        if not other.count:
            return
        if not self.count:
            self.count, self.minimum, self.maximum, self.mean, self.m2 = (
                other.count, other.minimum, other.maximum, other.mean, other.m2
            )
            return
        count = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / count
        self.mean += delta * other.count / count
        self.count = count
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)


@dataclass
class Histogram:
    edges: np.ndarray
    counts: np.ndarray
    below_range: int = 0
    above_range: int = 0

    @classmethod
    def empty(cls, edges: np.ndarray) -> Histogram:
        return cls(edges.copy(), np.zeros(len(edges) - 1, dtype=np.int64))

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().flatten()
        edges = torch.as_tensor(self.edges, device=values.device)
        # Calibration samples can contain 200 million pixels; bound the index buffers.
        for chunk in values.split(HISTOGRAM_CHUNK_VALUES):
            chunk = chunk.double()
            indices = (torch.bucketize(chunk, edges, right=True) - 1).clamp(0, len(self.counts) - 1)
            self.counts += torch.bincount(indices, minlength=len(self.counts)).cpu().numpy()
            self.below_range += int((chunk < edges[0]).sum().item())
            self.above_range += int((chunk > edges[-1]).sum().item())

    def merge(self, other: Histogram) -> None:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("Cannot merge histograms with different bin edges")
        self.counts += other.counts
        self.below_range += other.below_range
        self.above_range += other.above_range


@dataclass
class Distribution:
    moments: Moments
    histogram: Histogram

    @classmethod
    def empty(cls, edges: np.ndarray) -> Distribution:
        return cls(Moments(), Histogram.empty(edges))

    def update(self, values: torch.Tensor) -> None:
        self.moments.update(values)
        self.histogram.update(values)

    def merge(self, other: Distribution) -> None:
        self.moments.merge(other.moments)
        self.histogram.merge(other.histogram)


@dataclass
class PatchStats:
    patch_count: int = 0
    flat_patch_count: int = 0
    elevation_m: Moments = field(default_factory=Moments)
    elevation: Distribution = field(default_factory=lambda: Distribution.empty(ELEVATION_BINS))
    patch_means: Distribution = field(default_factory=lambda: Distribution.empty(ELEVATION_BINS))
    nonflat_patch_stds: Distribution = field(default_factory=lambda: Distribution.empty(PATCH_STD_BINS))
    downsampled_patch_stds: Distribution = field(default_factory=lambda: Distribution.empty(DOWNSAMPLED_PATCH_STD_BINS))
    nonflat_downsampled_patch_stds: Distribution = field(default_factory=lambda: Distribution.empty(DOWNSAMPLED_PATCH_STD_BINS))
    downsampled_to_original_std_ratios: Distribution = field(default_factory=lambda: Distribution.empty(STD_RATIO_BINS))
    patch_std_relationship: np.ndarray = field(default_factory=lambda: np.zeros(
        (len(PATCH_STD_BINS) - 1, len(DOWNSAMPLED_PATCH_STD_BINS) - 1), dtype=np.int64,
    ))
    reconstruction_residuals: Distribution = field(default_factory=lambda: Distribution.empty(RESIDUAL_BINS))
    training_residuals: Distribution = field(default_factory=lambda: Distribution.empty(RESIDUAL_BINS))

    def merge(self, other: PatchStats) -> None:
        self.patch_count += other.patch_count
        self.flat_patch_count += other.flat_patch_count
        self.elevation_m.merge(other.elevation_m)
        self.patch_std_relationship += other.patch_std_relationship
        for name in (
            "elevation", "patch_means", "nonflat_patch_stds",
            "downsampled_patch_stds", "nonflat_downsampled_patch_stds", "downsampled_to_original_std_ratios",
            "reconstruction_residuals", "training_residuals",
        ):
            getattr(self, name).merge(getattr(other, name))


@dataclass
class PatchMeanAnalysis:
    distribution: Distribution
    pdf: torch.Tensor


@dataclass
class CalibrationStats:
    residual_divisor_estimate: float | None
    downscale_patch_divisor_estimate: float | None
    existing_residual_divisor: float
    existing_downscale_patch_divisor: float
    residual_value_count: int
    downscale_value_count: int
    maximum_absolute_residual: float
    existing_normalized_residuals: Distribution | None
    candidate_normalized_residuals: Distribution | None
    candidate_maximum_to_stddev_ratio: float | None


@dataclass
class SampleStats:
    sample_count: int
    patch_means: PatchMeanAnalysis
    patch_ranges: Distribution
    range_percentiles: dict[str, float]
    calibration: CalibrationStats | None


@dataclass
class LevelStats:
    z_level: int
    patch_size: int
    source: str
    dataset: PatchStats = field(default_factory=PatchStats)
    sample: SampleStats | None = None


@dataclass
class AnalysisResults:
    metadata: dict
    levels: list[LevelStats]
    physical_total: PatchStats
    z10_source: PatchStats | None
    normalization_estimate_m: Moments | None


def fit_patch_mean_pdf(
    patch_means: torch.Tensor, *, grid_size: int = DEFAULT_GRID_SIZE,
) -> torch.Tensor:
    """Return a 2xN tensor containing a Gaussian-smoothed empirical PDF."""
    samples = patch_means.detach().flatten()
    sample_count = samples.numel()
    if not sample_count or grid_size < 2:
        raise ValueError("PDF fitting needs at least one sample and two grid points")
    stddev = samples.std(correction=0)
    bandwidth = torch.where(
        stddev > 0, stddev * sample_count ** (-0.2), torch.ones_like(stddev),
    )
    minimum, maximum = torch.aminmax(samples)
    minimum = minimum - 4.0 * bandwidth
    maximum = maximum + 4.0 * bandwidth
    grid_step = (maximum - minimum) / grid_size
    grid = minimum + (
        torch.arange(grid_size, device=samples.device, dtype=samples.dtype) + 0.5
    ) * grid_step
    bin_indices = ((samples - minimum) / grid_step).long().clamp(0, grid_size - 1)
    counts = torch.bincount(bin_indices, minlength=grid_size).to(samples.dtype)
    density = counts / (sample_count * grid_step)
    kernel_radius = max(1, int(torch.ceil(4.0 * bandwidth / grid_step).item()))
    kernel_positions = torch.arange(
        -kernel_radius, kernel_radius + 1, device=samples.device, dtype=samples.dtype,
    ) * grid_step
    kernel = torch.exp(-0.5 * (kernel_positions / bandwidth).square())
    kernel /= kernel.sum()
    density = F.conv1d(
        density.view(1, 1, -1), kernel.view(1, 1, -1), padding=kernel_radius,
    ).flatten()
    density /= density.sum() * grid_step
    return torch.stack((grid, density))


def evaluate_patch_mean_pdf(pdf: torch.Tensor, patch_means: torch.Tensor) -> torch.Tensor:
    """Interpolate PDF values, extending the endpoint densities as open tails."""
    grid, density = pdf
    grid_step = grid[1] - grid[0]
    positions = ((patch_means - grid[0]) / grid_step).clamp(0, grid.numel() - 1)
    lower_indices = positions.floor().long().clamp(0, grid.numel() - 2)
    upper_indices = lower_indices + 1
    interpolation = positions - lower_indices
    return density[lower_indices] * (1.0 - interpolation) + density[upper_indices] * interpolation


def evaluate_patch_pdf(pdf: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
    """Evaluate p(mean(patch)) for BCPP normalized-elevation patches."""
    return evaluate_patch_mean_pdf(pdf, patches.mean(dim=(-2, -1)))


def save_patch_mean_pdf(pdf: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pdf.cpu(), path)


def load_patch_mean_pdf(path: Path, *, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.load(path, map_location=device, weights_only=True)


def sample_elevation_values(
    base_dir: Path | str, *, sample_count: int = DEFAULT_SAMPLE_COUNT, seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Sample one raw-meter pixel from each of distinct randomly selected z10 patches."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    size = PATCH_SIZES[SYNTHETIC_SOURCE_Z_LEVEL]
    paths = _discover_shards(base_dir, SYNTHETIC_SOURCE_Z_LEVEL, size)
    counts = np.array([_shard_patch_count(path, size) for path in paths], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    if sample_count > offsets[-1]:
        raise ValueError(f"Requested {sample_count:,} patches but z10 contains only {offsets[-1]:,}")
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(int(offsets[-1]), size=sample_count, replace=False)
    pixel_ids = rng.integers(0, size * size, size=sample_count)
    shard_ids = np.searchsorted(offsets, sample_ids, side="right") - 1
    values = np.empty(sample_count, dtype=np.float32)
    for index in np.unique(shard_ids):
        positions = np.nonzero(shard_ids == index)[0]
        shard = np.memmap(paths[int(index)], dtype=np.float32, mode="r").reshape(-1, size * size)
        values[positions] = shard[sample_ids[positions] - offsets[index], pixel_ids[positions]]
    return values


def compute_normalization(
    base_dir: Path | str, *, sample_count: int = DEFAULT_SAMPLE_COUNT, seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    values = sample_elevation_values(base_dir, sample_count=sample_count, seed=seed)
    mean_m = float(np.mean(values, dtype=np.float64))
    std_m = float(np.std(values, dtype=np.float64))
    if std_m <= 0:
        raise ValueError("Sampled elevation values have zero standard deviation")
    return mean_m, std_m


def _shard_patch_count(path: Path, patch_size: int) -> int:
    patch_bytes = patch_size * patch_size * 4
    if path.stat().st_size % patch_bytes:
        raise ValueError(f"Shard contains a partial patch: {path}")
    return path.stat().st_size // patch_bytes


def _iter_raw_patch_batches(
    shard_paths: Iterable[Path], patch_size: int, batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for path in shard_paths:
        _shard_patch_count(path, patch_size)
        with path.open("rb") as shard_file:
            while True:
                values = np.fromfile(shard_file, dtype=np.float32, count=batch_size * patch_size**2)
                if not values.size:
                    break
                yield torch.from_numpy(values).reshape(-1, 1, patch_size, patch_size)


def _downsample_patches(patches: torch.Tensor) -> torch.Tensor:
    if patches.shape[-1] % 2:
        return patches.mean(dim=(2, 3), keepdim=True)
    return F.avg_pool2d(patches, 2)


def _pooled_upsample(patches: torch.Tensor) -> torch.Tensor:
    return F.interpolate(_downsample_patches(patches), size=patches.shape[-2:], mode="nearest")


def _upscale_deltas(
    real_patches: torch.Tensor, *, source_patches: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return packed target-minus-pooled-source residuals in training channel order."""
    source = real_patches if source_patches is None else source_patches
    return F.pixel_unshuffle(real_patches - _pooled_upsample(source), 2)


def _update_patch_stats(
    stats: PatchStats, patches: torch.Tensor, *, source_patches: torch.Tensor | None = None,
    raw_patches: torch.Tensor | None = None,
) -> None:
    stats.patch_count += len(patches)
    stats.elevation_m.update(denormalize_elevation(patches) if raw_patches is None else raw_patches)
    stats.elevation.update(patches)
    stats.patch_means.update(patches.mean(dim=(1, 2, 3)))
    nonflat = patches.flatten(1).amax(1) != patches.flatten(1).amin(1)
    stats.flat_patch_count += int((~nonflat).sum().item())
    original_stds = patches.double().std(dim=(1, 2, 3), correction=0)
    downsampled_stds = _downsample_patches(patches.double()).std(dim=(1, 2, 3), correction=0)
    stats.downsampled_patch_stds.update(downsampled_stds)
    varied = patches[nonflat]
    if len(varied):
        original_stds, downsampled_stds = original_stds[nonflat], downsampled_stds[nonflat]
        stats.nonflat_patch_stds.update(original_stds)
        stats.nonflat_downsampled_patch_stds.update(downsampled_stds)
        stats.downsampled_to_original_std_ratios.update(downsampled_stds / original_stds)
        original_bins = (torch.bucketize(
            original_stds, torch.as_tensor(PATCH_STD_BINS, device=patches.device), right=True,
        ) - 1).clamp(0, stats.patch_std_relationship.shape[0] - 1)
        downsampled_bins = (torch.bucketize(
            downsampled_stds, torch.as_tensor(DOWNSAMPLED_PATCH_STD_BINS, device=patches.device), right=True,
        ) - 1).clamp(0, stats.patch_std_relationship.shape[1] - 1)
        indices = original_bins * stats.patch_std_relationship.shape[1] + downsampled_bins
        stats.patch_std_relationship += torch.bincount(
            indices, minlength=stats.patch_std_relationship.size,
        ).reshape(stats.patch_std_relationship.shape).cpu().numpy()
        stats.reconstruction_residuals.update(_pooled_upsample(varied) - varied)
    source = patches if source_patches is None else source_patches
    stats.training_residuals.update(patches - _pooled_upsample(source))


@torch.inference_mode()
def analyze_patch_batches(
    batches: Iterable[torch.Tensor], *, z_level: int, patch_size: int,
    source_batches: Iterable[torch.Tensor] | None = None,
) -> LevelStats:
    """Analyze normalized B1HW batches without fitting PDFs or writing artifacts."""
    stats = LevelStats(z_level, patch_size, "normalized batches")
    sources = iter(source_batches) if source_batches is not None else None
    for patches in batches:
        source = next(sources, None) if sources is not None else None
        if sources is not None and source is None:
            raise ValueError("Source and target batch counts differ")
        if source is not None and source.shape != patches.shape:
            raise ValueError("Source and target batches must have the same shape")
        _update_patch_stats(stats.dataset, patches, source_patches=source)
    if sources is not None and next(sources, None) is not None:
        raise ValueError("Source and target batch counts differ")
    return stats


def _sample_distribution(values: torch.Tensor) -> Distribution:
    moments = Moments()
    moments.update(values)
    if moments.minimum == moments.maximum:
        radius = max(0.5, abs(moments.minimum) * 1e-6)
        edges = np.linspace(moments.minimum - radius, moments.maximum + radius, 257)
    else:
        width = 3.49 * moments.stddev / moments.count ** (1 / 3)
        bins = min(4096, max(1, math.ceil((moments.maximum - moments.minimum) / width)))
        edges = np.linspace(moments.minimum, moments.maximum, bins + 1)
    histogram = Histogram.empty(edges)
    histogram.update(values)
    return Distribution(moments, histogram)


@torch.inference_mode()
def analyze_patch_means(
    patch_means: torch.Tensor, *, grid_size: int = DEFAULT_GRID_SIZE,
) -> PatchMeanAnalysis:
    """Fit already-normalized means; reusable for generated input-map crops."""
    if not patch_means.numel():
        raise ValueError("Patch-mean analysis needs at least one sample")
    return PatchMeanAnalysis(
        _sample_distribution(patch_means), fit_patch_mean_pdf(patch_means, grid_size=grid_size).cpu(),
    )


def _transformed_stddev(values: torch.Tensor, divisor: float) -> float:
    return torch.asinh(values / divisor).std(correction=0).item()


def _find_divisor(values: torch.Tensor) -> float | None:
    if not values.numel():
        return None
    initial = values.std(correction=0).item()
    if initial <= 0:
        return None
    lower = upper = initial
    # A finite bracket also handles degenerate populations without an endless search.
    for _ in range(128):
        if _transformed_stddev(values, lower) >= TARGET_TRANSFORMED_STDDEV:
            break
        upper, lower = lower, lower / 2
    else:
        return None
    for _ in range(128):
        if _transformed_stddev(values, upper) <= TARGET_TRANSFORMED_STDDEV:
            break
        lower, upper = upper, upper * 2
    else:
        return None
    for _ in range(DIVISOR_SEARCH_STEPS):
        divisor = (lower + upper) / 2
        if _transformed_stddev(values, divisor) < TARGET_TRANSFORMED_STDDEV:
            upper = divisor
        else:
            lower = divisor
    return (lower + upper) / 2


@torch.inference_mode()
def _analyze_sample(
    patches: torch.Tensor, z_level: int, *, source_patches: torch.Tensor | None = None,
) -> SampleStats:
    means = analyze_patch_means(patches.mean(dim=(1, 2, 3)))
    ranges = patches.flatten(1).amax(1) - patches.flatten(1).amin(1)
    quantiles = torch.quantile(ranges, torch.tensor((0.5, 0.75, 0.9, 0.95, 0.99), device=patches.device))
    calibration = None
    if z_level:
        residuals = _upscale_deltas(patches, source_patches=source_patches).flatten()
        downscaled = F.avg_pool2d(patches, 2)
        centered = downscaled - downscaled.mean(dim=(1, 2, 3), keepdim=True)
        nonflat = downscaled.flatten(1).amax(1) != downscaled.flatten(1).amin(1)
        downscale_values = centered[nonflat].flatten()
        estimate = _find_divisor(residuals) if torch.any(ranges > 0).item() else None
        downscale_estimate = _find_divisor(downscale_values)
        current = UPSCALE_STATS[z_level]
        existing = _sample_distribution(torch.asinh(residuals / current.residual_divisor))
        candidate = _sample_distribution(torch.asinh(residuals / estimate)) if estimate else None
        ratio = None
        if candidate and candidate.moments.stddev:
            ratio = max(abs(candidate.moments.minimum), abs(candidate.moments.maximum)) / candidate.moments.stddev
        calibration = CalibrationStats(
            estimate, downscale_estimate, current.residual_divisor, current.downscale_patch_divisor,
            residuals.numel(), downscale_values.numel(), residuals.abs().max().item(),
            existing, candidate, ratio,
        )
    return SampleStats(
        len(patches), means, _sample_distribution(ranges),
        dict(zip(("50%", "75%", "90%", "95%", "99%"), quantiles.tolist())), calibration,
    )


@torch.inference_mode()
def analyze_patches(
    *, base_dir: Path = DEFAULT_ETOPO_BASE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_size: int = DEFAULT_SAMPLE_SIZE, batch_size: int = DEFAULT_BATCH_SIZE,
    normalization_sample_count: int = DEFAULT_SAMPLE_COUNT, seed: int = DEFAULT_SEED,
    device: torch.device | str = "cuda", z_levels: Iterable[int] | None = None,
) -> AnalysisResults:
    """Scan all selected physical levels and synthesize z11-z20 from selected z10."""
    if min(sample_size, batch_size, normalization_sample_count) <= 0:
        raise ValueError("Sample sizes and batch size must be positive")
    base_dir, output_dir = Path(base_dir), Path(output_dir)
    available = [z for z in range(11) if (base_dir / f"z{z}").is_dir()]
    selected = available if z_levels is None else sorted(set(z_levels))
    if not selected:
        raise FileNotFoundError(f"No z-level datasets found in {base_dir}")
    if any(z < 0 or z > 10 for z in selected):
        raise ValueError("Select physical z-levels between 0 and 10")
    paths_by_level = {z: _discover_shards(base_dir, z, PATCH_SIZES[z]) for z in selected}
    omitted = [z for z in range(11) if z not in selected]
    print(f"Physical levels: {selected}; omitted: {omitted}", flush=True)
    levels = []
    physical_total = PatchStats()
    z10_source = None
    for z in selected:
        size = PATCH_SIZES[z]
        scales = (1.0, *SYNTHETIC_STD_SCALES) if z == 10 else (1.0,)
        group = [LevelStats(
            z + index, size, "physical shards" if index == 0 else f"z10 deviations x {scale}; source z{z + index - 1}",
        ) for index, scale in enumerate(scales)]
        sample_batches = []
        sampled_count = 0
        print(f"Scanning z{z} ({size}x{size})...", flush=True)
        for batch_index, raw_batch in enumerate(_iter_raw_patch_batches(paths_by_level[z], size, batch_size)):
            device_raw = raw_batch.to(device)
            patches = normalize_elevation(device_raw)
            if sampled_count < sample_size:
                sample = patches[:sample_size - sampled_count].cpu().clone()
                sample_batches.append(sample)
                sampled_count += len(sample)
            source = patches
            for scale, level in zip(scales, group):
                target = patches if scale == 1 else scale_patch_deviations(patches, scale)
                _update_patch_stats(
                    level.dataset, target, source_patches=source,
                    raw_patches=device_raw if scale == 1 else None,
                )
                source = target
            if (batch_index + 1) % 128 == 0:
                print(f"  scanned {group[0].dataset.patch_count:,} physical patches", flush=True)
        if not sampled_count:
            raise ValueError(f"No patches found for z{z}")
        physical_total.merge(group[0].dataset)
        if z == 10:
            z10_source = group[0].dataset
        sample = torch.cat(sample_batches).to(device)
        del sample_batches
        source_sample = sample
        for scale, level in zip(scales, group):
            target = sample if scale == 1 else scale_patch_deviations(sample, scale)
            level.sample = _analyze_sample(target, level.z_level, source_patches=source_sample)
            print(
                f"  z{level.z_level}: {level.dataset.patch_count:,} patches; "
                f"{level.dataset.flat_patch_count:,} flat; {level.sample.sample_count:,} fit samples",
                flush=True,
            )
            source_sample = target
        levels.extend(group)
        del sample, source_sample, target
    estimate = None
    estimate_count = 0
    if z10_source:
        estimate_count = min(normalization_sample_count, z10_source.patch_count)
        values = sample_elevation_values(base_dir, sample_count=estimate_count, seed=seed)
        estimate = Moments()
        estimate.update(torch.from_numpy(values))
    results = AnalysisResults({
        "base_dir": str(base_dir), "physical_levels": selected, "omitted_physical_levels": omitted,
        "device": str(device), "batch_size": batch_size,
        "sample_size_requested": sample_size, "fit_sampling": "first N patches in sorted shard order",
        "normalization_sample_count_requested": normalization_sample_count,
        "normalization_sample_count_used": estimate_count, "seed": seed,
        "normalization_sampling": "one random pixel per distinct randomly selected z10 patch",
        "existing_elevation_mean_m": ELEVATION_MEAN_M, "existing_elevation_std_m": ELEVATION_STD_M,
        "normalization_policy": "existing constants; estimates are review-only",
        "population_std": "ddof=0",
        "downsampled_patch_stds": "std(pool(target)); all patches; 2x2 average pooling, or all nine z0 pixels to 1x1",
        "nonflat_downsampled_patch_stds": "std(pool(target)); original non-flat patches only; used for overlaid comparisons",
        "downsampled_to_original_std_ratios": "std(pool(target)) / std(target); original non-flat patches only; zero pooled std included",
        "physical_total_population": "selected physical shards; pixel-weighted elevation and patch-weighted patch metrics",
        "reconstruction_residuals": "upsample(pool(target)) - target; non-flat patches only; z0 pools all nine pixels",
        "training_residuals": "target - upsample(pool(source)); all patches; source is target for physical levels and previous synthetic level otherwise",
        "histogram_edge_policy": "out-of-range values accumulate in edge bins; counts recorded separately",
    }, levels, physical_total, z10_source, estimate)
    output_dir.mkdir(parents=True, exist_ok=True)
    for level in levels:
        save_patch_mean_pdf(level.sample.patch_means.pdf, output_dir / f"z{level.z_level}_patch_mean_pdf.pt")
    write_analysis_report(results, output_dir)
    _print_constants(results)
    return results


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_distribution(
    axis, distribution: Distribution, *, label: str, color: str = "#206e82", percent: bool = False,
) -> None:
    histogram = distribution.histogram
    count = histogram.counts.sum()
    if not count:
        axis.text(0.5, 0.5, "No observations", ha="center", va="center", transform=axis.transAxes)
        return
    occupied = np.flatnonzero(histogram.counts)
    start, stop = occupied[0], occupied[-1] + 1
    edges = histogram.edges[start:stop + 1]
    counts = histogram.counts[start:stop]
    # Log-spaced bins near zero have tiny widths; show their mass, not a density spike.
    heights = 100 * counts / count if percent else counts / (count * np.diff(edges))
    axis.stairs(heights, edges, alpha=0.65, color=color, label=label)
    axis.set_ylabel("Observations per bin (%)" if percent else "Density")
    axis.grid(alpha=0.2)


def _set_distribution_scale(axis, distribution: Distribution, xscale: str) -> None:
    """Use occupied bins from this population, while keeping shared bins for aggregation."""
    histogram = distribution.histogram
    occupied = np.flatnonzero(histogram.counts)
    if not len(occupied):
        return
    low, high = histogram.edges[occupied[0]], histogram.edges[occupied[-1] + 1]
    if xscale == "symlog":
        extent = max(abs(low), abs(high))
        linthresh = max(distribution.moments.stddev / 10, extent * 1e-4)
        axis.set_xscale("symlog", linthresh=linthresh)
        axis.set_xlim(low, high)
        axis.set_xticks([value for value in axis.get_xticks() if value == 0 or abs(value) >= linthresh])
    else:
        axis.set_xscale(xscale)
    axis.set_xlim(low, high)
    if xscale != "linear":
        axis.set_yscale("log")


def write_patch_mean_chart(axis, analysis: PatchMeanAnalysis, *, title: str) -> None:
    """Draw one empirical histogram and smoothed PDF using normalized-elevation means."""
    _draw_distribution(axis, analysis.distribution, label="Empirical histogram")
    grid, density = analysis.pdf.numpy()
    axis.plot(grid, density, color="#b34f43", linewidth=1.6, label="Gaussian-smoothed PDF")
    axis.axvline(analysis.distribution.moments.mean, color="#43835d", linestyle="--", label="Mean")
    axis.set(title=title, xlabel="Mean normalized elevation")
    axis.legend()


def save_patch_mean_chart(analysis: PatchMeanAnalysis, path: Path, *, title: str) -> None:
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(10, 6))
    write_patch_mean_chart(axis, analysis, title=title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_patch_std_comparison(axis, stats: PatchStats) -> None:
    count = stats.nonflat_patch_stds.moments.count
    axis.set(title="Original and downsampled patch std", xlabel="Std in normalized elevation",
             ylabel="Non-flat patches per bin (%)", xscale="log", yscale="log")
    if not count:
        axis.text(0.5, 0.5, "No non-flat patches", ha="center", va="center", transform=axis.transAxes)
        return
    original_counts = stats.nonflat_patch_stds.histogram.counts
    downsampled_counts = stats.nonflat_downsampled_patch_stds.histogram.counts[1:]
    occupied = np.flatnonzero(original_counts + downsampled_counts)
    start, stop = occupied[0], occupied[-1] + 1
    edges = PATCH_STD_BINS[start:stop + 1]
    for counts, label, color, linestyle in (
        (original_counts, "Original", "#206e82", "-"),
        (downsampled_counts, "Downsampled", "#b34f43", "--"),
    ):
        axis.stairs(100 * counts[start:stop] / count, edges, color=color, linestyle=linestyle, label=label)
    axis.set_xlim(edges[0], edges[-1])
    plotted_counts = np.concatenate((original_counts[start:stop], downsampled_counts[start:stop]))
    positive_percentages = 100 * plotted_counts[plotted_counts > 0] / count
    axis.set_ylim(positive_percentages.min() / 2, min(105, positive_percentages.max() * 2))
    below_range = stats.nonflat_downsampled_patch_stds.histogram.counts[0]
    if below_range:
        axis.text(0.03, 0.03, f"Downsampled std < {PATCH_STD_BINS[0]:g}: {100 * below_range / count:.3g}%",
                  transform=axis.transAxes, fontsize=9)
    axis.grid(alpha=0.2)
    axis.legend()


def write_patch_std_trends(axis, levels: list[LevelStats], *, log_scale: bool) -> None:
    z_levels = [level.z_level for level in levels]
    omitted_zero = False
    for attr, label, color, linestyle in (
        ("nonflat_patch_stds", "Non-flat patch std", "#206e82", "-"),
        ("nonflat_downsampled_patch_stds", "Downsampled patch std", "#b34f43", "--"),
    ):
        values = [getattr(level.dataset, attr).moments.mean for level in levels]
        if log_scale:
            omitted_zero |= any(value == 0 for value in values)
            values = [value if value is not None and value > 0 else None for value in values]
        axis.plot(z_levels, values, marker="o", color=color, linestyle=linestyle, label=label)
    axis.set(title="Original vs downsampled patch std", xlabel="z-level", xticks=z_levels,
             ylabel="Mean per-patch std (normalized elevation)", yscale="log" if log_scale else "linear")
    if omitted_zero:
        axis.text(0.5, 0.03, "Zero std omitted on log axis", transform=axis.transAxes, ha="center", fontsize=9)
    axis.grid(alpha=0.2)
    axis.legend()


def write_downsampling_chart(stats: PatchStats, *, title: str):
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import NullFormatter
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    write_patch_std_comparison(axes[0], stats)
    _draw_distribution(axes[1], stats.downsampled_to_original_std_ratios, label="Original non-flat patches", percent=True)
    _set_distribution_scale(axes[1], stats.downsampled_to_original_std_ratios, "linear")
    axes[1].set(title="Std retained after downsampling", xlabel="Downsampled std / original std")
    axes[1].set_xlim(0, 1)
    occupied = np.argwhere(stats.patch_std_relationship)
    if len(occupied):
        first, last = occupied.min(axis=0), occupied.max(axis=0) + 1
        original_edges = PATCH_STD_BINS[first[0]:last[0] + 1]
        downsampled_edges = DOWNSAMPLED_PATCH_STD_BINS[first[1]:last[1] + 1]
        counts = stats.patch_std_relationship[first[0]:last[0], first[1]:last[1]]
        image = axes[2].pcolormesh(
            original_edges, downsampled_edges, np.ma.masked_equal(counts.T, 0),
            norm=LogNorm(vmin=1, vmax=max(2, counts.max())), cmap="viridis", rasterized=True,
        )
        figure.colorbar(image, ax=axes[2], label="Patches per bin")
        axes[2].plot(original_edges, original_edges, linestyle="--", color="#b34f43", label="Equal std")
        axes[2].set_xscale("log")
        axes[2].set_yscale("symlog", linthresh=PATCH_STD_BINS[0])
        axes[2].set_xlim(original_edges[0], original_edges[-1])
        upper = max(original_edges[-1], downsampled_edges[-1])
        axes[2].set_ylim(downsampled_edges[0], upper)
        original_ticks = np.geomspace(original_edges[0], original_edges[-1], 3)
        downsampled_ticks = np.geomspace(max(downsampled_edges[0], PATCH_STD_BINS[0]), upper, 3)
        if downsampled_edges[0] == 0:
            downsampled_ticks = np.concatenate(([0], downsampled_ticks))
        axes[2].set_xticks(original_ticks, labels=[f"{value:.3g}" for value in original_ticks])
        axes[2].set_yticks(downsampled_ticks, labels=[f"{value:.3g}" for value in downsampled_ticks])
        axes[2].xaxis.set_minor_formatter(NullFormatter())
        axes[2].yaxis.set_minor_formatter(NullFormatter())
        axes[2].legend()
    else:
        axes[2].text(0.5, 0.5, "No non-flat patches", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set(title="Original vs downsampled std (non-flat)", xlabel="Original std", ylabel="Downsampled std")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def _format(value) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _table_page(title: str, headers: list[str], rows: list[list[str]], notes: str):
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(15, 8.5))
    axis.axis("off")
    figure.text(0.035, 0.94, title, fontsize=19, weight="bold")
    figure.text(0.035, 0.86, notes, fontsize=9, va="top")
    first_column_width = max(
        1 / len(headers), min(0.32, max(len(row[0]) for row in [headers, *rows]) * 0.008),
    )
    other_column_width = (1 - first_column_width) / (len(headers) - 1)
    table = axis.table(
        cellText=rows, colLabels=headers, cellLoc="right", colLoc="right",
        colWidths=[first_column_width, *[other_column_width] * (len(headers) - 1)],
        bbox=[0, 0.06, 1, 0.7],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5da")
        cell.set_facecolor("#e5eff1" if row == 0 else ("#f5f7f8" if row % 2 == 0 else "white"))
    return figure


def write_analysis_report(results: AnalysisResults, output_dir: Path) -> None:
    """Render per-level PNG charts and a multipage report from the same figures."""
    from matplotlib.backends.backend_pdf import PdfPages
    plt = _pyplot()
    output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_dir / "patch_analysis.pdf") as pdf:
        def save(figure, name):
            pdf.savefig(figure)
            figure.savefig(output_dir / f"{name}.png", dpi=140)
            plt.close(figure)

        summaries = [("physical_dataset", "Pooled physical z0-z10", results.physical_total)]
        if results.z10_source is not None:
            summaries.append(("z10_source_dataset", "z10 source dataset (full scan)", results.z10_source))
        rows = []
        for _, label, stats in summaries:
            for units, moments in (("meters", stats.elevation_m), ("normalized", stats.elevation.moments)):
                rows.append([label, units, f"{stats.patch_count:,}", f"{moments.count:,}", *[
                    _format(value) for value in (moments.minimum, moments.mean, moments.stddev, moments.maximum)
                ]])
        estimate = results.normalization_estimate_m
        if estimate:
            rows.append(["z10 random-pixel estimate", "meters", f"{estimate.count:,}", f"{estimate.count:,}", *[
                _format(value) for value in (estimate.minimum, estimate.mean, estimate.stddev, estimate.maximum)
            ]])
        metadata = results.metadata
        notes = (
            f"Source: {metadata['base_dir']}\n"
            f"Existing elevation mean/std (m): {ELEVATION_MEAN_M:.9f} / {ELEVATION_STD_M:.9f}. Estimates are review-only.\n"
            f"Physical levels: {metadata['physical_levels']}; omitted: {metadata['omitted_physical_levels']}. "
            f"Seed: {metadata['seed']}; fit sample limit: {metadata['sample_size_requested']:,} (first N).\n"
            "Population std (ddof=0). Pooled totals exclude synthetic levels and describe stored multiscale patches, not Earth surface area."
        )
        save(_table_page("Whole-dataset elevation statistics", [
            "Population", "Units", "Patches", "Pixels / samples", "Min", "Mean", "Std", "Max",
        ], rows, notes), "dataset_summary")
        for name, label, stats in summaries:
            metrics = (
                ("elevation", "Elevation (pixels)"), ("patch_means", "Patch means (all patches)"),
                ("nonflat_patch_stds", "Patch std (non-flat patches)"),
                ("downsampled_patch_stds", "Downsampled patch std (all patches)"),
                ("downsampled_to_original_std_ratios", "Downsampled / original std (non-flat)"),
                ("reconstruction_residuals", "Reconstruction residuals (non-flat)"),
                ("training_residuals", "Training residuals (all patches)"),
            )
            rows = []
            for attr, title in metrics:
                moments = getattr(stats, attr).moments
                rows.append([title, f"{moments.count:,}", *[_format(value) for value in (
                    moments.minimum, moments.mean, moments.stddev, moments.maximum,
                )]])
            save(_table_page(f"{label}: full population statistics", [
                "Metric (normalized elevation; ratios unitless)", "Observations", "Min", "Mean", "Std", "Max",
            ], rows, f"Patches: {stats.patch_count:,}; flat: {stats.flat_patch_count:,}. "
                "Each metric is weighted by its observation count.\n"
                "Recon = upsample(pool(target)) - target; Train = target - upsample(pool(source))."),
                 f"{name}_metrics")
            figure, axes = plt.subplots(2, 4, figsize=(18, 8.5))
            for axis, (attr, title) in zip(axes.flat, metrics):
                if attr == "nonflat_patch_stds":
                    write_patch_std_comparison(axis, stats)
                    continue
                distribution = getattr(stats, attr)
                xscale = "log" if attr == "nonflat_patch_stds" else (
                    "symlog" if "residuals" in attr or attr == "downsampled_patch_stds" else "linear"
                )
                _draw_distribution(axis, distribution, label="Full scan", percent=xscale != "linear")
                axis.set(title=title, xlabel="Std ratio" if attr == "downsampled_to_original_std_ratios" else "Normalized elevation")
                _set_distribution_scale(axis, distribution, xscale)
                if distribution.moments.count:
                    axis.legend()
            axes.flat[-1].axis("off")
            figure.suptitle(f"{label}: full-dataset histograms (edge bins include out-of-range values)")
            figure.tight_layout()
            save(figure, f"{name}_histograms")
            save(write_downsampling_chart(stats, title=f"{label}: downsampling variability"), f"{name}_downsampling")
        levels = results.levels
        for start in range(0, len(levels), 11):
            group = levels[start:start + 11]
            labels = [f"z{level.z_level}" for level in group]
            for units, attr in (("normalized elevation", "elevation"), ("meters", "elevation_m")):
                rows = []
                for label, level in zip(labels, group):
                    metric = getattr(level.dataset, attr)
                    moments = metric.moments if isinstance(metric, Distribution) else metric
                    rows.append([label, f"{level.patch_size}x{level.patch_size}", f"{level.dataset.patch_count:,}",
                                 f"{level.dataset.flat_patch_count:,}", *[_format(value) for value in (
                                     moments.minimum, moments.mean, moments.stddev, moments.maximum,
                                 )]])
                save(_table_page(f"Per-level elevation statistics ({units})", [
                    "Level", "Patch", "Patches", "Flat", "Min", "Mean", "Std", "Max",
                ], rows, "Full scans include flat patches. z11-z20 are scaled z10 copies; they do not enter physical totals."),
                     f"level_elevation_{attr}_{start}")
            rows = []
            for label, level in zip(labels, group):
                stats = level.dataset
                std = stats.nonflat_patch_stds.moments
                reconstruction = stats.reconstruction_residuals.moments
                training = stats.training_residuals.moments
                rows.append([label, f"{std.count:,}", *[_format(value) for value in (
                    std.minimum, std.mean, std.maximum, reconstruction.minimum,
                    reconstruction.maximum, training.minimum, training.maximum,
                )]])
            save(_table_page("Patch variability and residual extrema (normalized elevation)", [
                "Level", "Non-flat patches", "Std min", "Std mean", "Std max", "Recon min", "Recon max", "Train min", "Train max",
            ], rows, "Recon = upsample(pool(target)) - target; non-flat patches only.\n"
                "Train = target - upsample(pool(source)); all patches. Synthetic source is previous level. z0 pools 3x3 to 1x1."),
                 f"level_variability_{start}")
            rows = []
            for label, level in zip(labels, group):
                stats = level.dataset
                pooled = stats.downsampled_patch_stds.moments
                ratio = stats.downsampled_to_original_std_ratios.moments
                rows.append([label, f"{pooled.count:,}", *[_format(value) for value in (
                    pooled.minimum, pooled.mean, pooled.maximum,
                )], f"{ratio.count:,}", *[_format(value) for value in (
                    ratio.minimum, ratio.mean, ratio.maximum,
                )]])
            save(_table_page("Downsampled patch variability and std retention", [
                "Level", "Pooled patches", "Std min", "Std mean", "Std max", "Ratio patches", "Ratio min", "Ratio mean", "Ratio max",
            ], rows, "Population std (ddof=0), in normalized elevation. Pool each target using 2x2 averages; z0 pools 3x3 to 1x1.\n"
                "Ratios are per-patch pooled std / original std, excluding original flat patches; zero pooled std is included."),
                 f"level_downsampling_{start}")
            rows = []
            for label, level in zip(labels, group):
                sample = level.sample
                c = sample.calibration
                rows.append([label, f"{sample.sample_count:,}", *[_format(value) for value in (
                    c.existing_residual_divisor if c else 0.0,
                    c.residual_divisor_estimate if c else None,
                    c.existing_downscale_patch_divisor if c else 0.0,
                    c.downscale_patch_divisor_estimate if c else None,
                    c.candidate_normalized_residuals.moments.stddev if c and c.candidate_normalized_residuals else None,
                    c.candidate_maximum_to_stddev_ratio if c else None,
                )]])
            save(_table_page("Sampled divisor estimates for review", [
                "Level", "Fit patches", "Existing residual", "Estimate residual", "Existing downscale", "Estimate downscale", "Candidate std", "Max / std",
            ], rows, "Existing constants remain active. Candidate diagnostics use fitted divisors only for review.\n"
                "Downscale fits exclude spatially constant pooled patches; residual fits include flat patches. N/A = no valid fit."),
                 f"calibration_{start}")
        for log_scale in (False, True):
            figure, axes = plt.subplots(2, 2, figsize=(13, 9))
            z = [level.z_level for level in levels]
            write_patch_std_trends(axes[0, 0], levels, log_scale=log_scale)
            for name, color in (("minimum", "#43835d"), ("maximum", "#b34f43")):
                axes[0, 1].plot(z, [getattr(level.dataset.training_residuals.moments, name) for level in levels],
                                marker="o", color=color, label=name)
            axes[1, 0].plot(z, [level.dataset.patch_count for level in levels], marker="o", label="All patches")
            axes[1, 0].plot(z, [level.dataset.flat_patch_count for level in levels], marker="o", label="Flat patches")
            axes[1, 1].plot(z, [level.sample.patch_ranges.moments.mean for level in levels], marker="o", label="Mean range")
            for axis, title in zip(axes.flat, ("Original vs downsampled patch std", "Training residual extrema", "Patch counts", "Sampled patch ranges")):
                axis.set(title=title, xlabel="z-level", xticks=z)
                axis.grid(alpha=0.2)
                axis.legend()
                if log_scale and axis is not axes[0, 0]:
                    axis.set_yscale("symlog", linthresh=1 if axis is axes[1, 0] else 1e-4)
            figure.suptitle("Cross-level trends: " + ("log scales" if log_scale else "linear"))
            figure.tight_layout()
            save(figure, "trends_log" if log_scale else "trends_linear")
        figure, axis = plt.subplots(figsize=(10, 6))
        write_patch_std_trends(axis, levels, log_scale=True)
        figure.tight_layout()
        save(figure, "patch_std_comparison_log")
        for level in levels:
            label = f"z{level.z_level}"
            sample = level.sample
            save(write_downsampling_chart(level.dataset, title=f"{label}: full-scan downsampling variability"),
                 f"{label}_downsampling")
            figure, axis = plt.subplots(figsize=(10, 6))
            write_patch_mean_chart(axis, sample.patch_means, title=f"{label} patch means ({sample.sample_count:,} sampled patches)")
            figure.tight_layout()
            save(figure, f"{label}_patch_mean_histogram")
            figure, axis = plt.subplots(figsize=(10, 6))
            _draw_distribution(axis, sample.patch_ranges, label="Patch ranges")
            for percentile, value in sample.range_percentiles.items():
                axis.axvline(value, linestyle=":", label=f"{percentile}: {value:.5g}")
            axis.set(title=f"{label} sampled patch value ranges", xlabel="max(patch) - min(patch)")
            _set_distribution_scale(axis, sample.patch_ranges, "linear")
            axis.set_xlim(left=max(0, axis.get_xlim()[0]))
            axis.legend()
            figure.tight_layout()
            save(figure, f"{label}_patch_value_range_histogram")
            c = sample.calibration
            if c:
                figure, axis = plt.subplots(figsize=(10, 6))
                _draw_distribution(axis, c.existing_normalized_residuals, label=f"Existing divisor: {c.existing_residual_divisor:.6g}")
                if c.candidate_normalized_residuals:
                    _draw_distribution(axis, c.candidate_normalized_residuals, color="#b34f43",
                                       label=f"Candidate divisor (review): {c.residual_divisor_estimate:.6g}")
                axis.set(title=f"{label} sampled asinh training residuals", xlabel="asinh(residual / divisor)")
                axis.legend()
                figure.tight_layout(pad=2)
                save(figure, f"{label}_upscale_delta_histogram")
            for attr, title, xscale in (
                ("nonflat_patch_stds", "Non-flat patch standard deviations", "log"),
                ("reconstruction_residuals", "Reconstruction residuals (non-flat patches)", "symlog"),
                ("training_residuals", "Training residuals (all patches)", "symlog"),
            ):
                distribution = getattr(level.dataset, attr)
                figure, axis = plt.subplots(figsize=(10, 6))
                if attr == "nonflat_patch_stds":
                    write_patch_std_comparison(axis, level.dataset)
                    axis.set_title(f"{label}: original and downsampled patch std")
                    figure.tight_layout()
                    save(figure, f"{label}_{attr}_histogram")
                    continue
                _draw_distribution(axis, distribution, label=f"{distribution.moments.count:,} observations", percent=True)
                axis.set(title=f"{label} {title}", xlabel="Normalized elevation")
                _set_distribution_scale(axis, distribution, xscale)
                moments = distribution.moments
                if moments.count:
                    axis.legend()
                axis.text(0.02, 0.97, "Full scan; out-of-range values included in edge bins", transform=axis.transAxes, va="top", fontsize=8)
                figure.tight_layout()
                save(figure, f"{label}_{attr}_histogram")
    for index in (0, 1):
        for kind in ("metrics", "histograms"):
            (output_dir / f"dataset_{kind}_{index}.png").unlink(missing_ok=True)
    print(f"Wrote {output_dir / 'patch_analysis.pdf'} and PNG charts", flush=True)


def _print_constants(results: AnalysisResults) -> None:
    estimate = results.normalization_estimate_m
    print("Candidate constants for review only (existing normalization was used):")
    if estimate:
        print(f"ELEVATION_MEAN_M: float = {estimate.mean:.9f}")
        print(f"ELEVATION_STD_M: float = {estimate.stddev:.9f}")
    else:
        print("Universal normalization estimate unavailable: z10 was not selected")
    print("UPSCALE_STAT_ROWS = [")
    print("    [0.0, 0.0, 0, None, None, None, None],  # z0 sentinel")
    by_level = {level.z_level: level for level in results.levels}
    for z in range(1, len(UPSCALE_STATS)):
        level = by_level.get(z)
        if level is None:
            stats = UPSCALE_STATS[z]
            row = [getattr(stats, item.name) for item in fields(stats) if item.name != "z_level"]
            note = "existing row; not analyzed"
        else:
            c = level.sample.calibration
            values = level.dataset.elevation.moments
            residuals = level.dataset.training_residuals.moments
            row = [c.residual_divisor_estimate, c.downscale_patch_divisor_estimate,
                   level.dataset.patch_count, values.minimum, values.maximum, residuals.minimum, residuals.maximum]
            note = level.source
        print(f"    {row!r},  # z{z}: {note}")
    print("]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_ETOPO_BASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--normalization-sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--z-levels", type=int, nargs="+")
    analyze_patches(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

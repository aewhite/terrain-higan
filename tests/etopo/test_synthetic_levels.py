import numpy as np
import torch
import torch.nn.functional as F

import terrain_higan.etopo.analyze_patches as analysis

from terrain_higan.etopo.analyze_patches import (
    analyze_patch_batches,
    _upscale_deltas,
)
from terrain_higan.etopo.normalization import (
    SYNTHETIC_STD_SCALES,
    normalize_elevation,
    scale_patch_deviations,
)


def test_synthetic_scales_preserve_patch_mean_and_scale_std() -> None:
    raw = torch.linspace(-8000.0, 9000.0, 200).reshape(2, 1, 10, 10)
    patches = normalize_elevation(raw)
    means = patches.mean(dim=(1, 2, 3))
    stddevs = patches.std(dim=(1, 2, 3), correction=0)

    for scale in SYNTHETIC_STD_SCALES:
        flattened = scale_patch_deviations(patches, scale)
        torch.testing.assert_close(flattened.mean(dim=(1, 2, 3)), means, rtol=0, atol=1e-6)
        torch.testing.assert_close(
            flattened.std(dim=(1, 2, 3), correction=0), stddevs * scale,
            rtol=1e-5, atol=1e-6,
        )


def test_dataset_stats_for_synthetic_levels_match_transformed_patches(tmp_path) -> None:
    raw = np.arange(300, dtype=np.float32).reshape(3, 1, 10, 10) * 100 - 8000
    shard_path = tmp_path / "shard-0000"
    shard_path.write_bytes(raw.tobytes())
    scales = (1.0, *SYNTHETIC_STD_SCALES)

    normalized = normalize_elevation(torch.from_numpy(raw))
    targets = [normalized if scale == 1.0 else scale_patch_deviations(normalized, scale) for scale in scales]
    stats = [analyze_patch_batches(
        [target], z_level=10 + index, patch_size=10, source_batches=[source],
    ).dataset for index, (target, source) in enumerate(zip(targets, [normalized, *targets[:-1]]))]
    assert len(stats) == 11
    source = normalized
    for scale, level_stats in zip(scales, stats):
        patches = normalized if scale == 1.0 else scale_patch_deviations(normalized, scale)
        pooled_source = F.avg_pool2d(source, 2)
        upscaled_source = pooled_source.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        residuals = patches - upscaled_source
        assert level_stats.patch_count == 3
        assert level_stats.elevation.moments.minimum == patches.min().item()
        assert level_stats.elevation.moments.maximum == patches.max().item()
        assert level_stats.training_residuals.moments.minimum == residuals.min().item()
        assert level_stats.training_residuals.moments.maximum == residuals.max().item()
        source = patches


def test_synthetic_residuals_use_previous_level_source() -> None:
    patches = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    source = patches
    for scale in SYNTHETIC_STD_SCALES:
        target = scale_patch_deviations(patches, scale)
        upscaled_source = F.interpolate(F.avg_pool2d(source, 2), scale_factor=2, mode="nearest")
        expected = F.pixel_unshuffle(target - upscaled_source, 2)
        actual = _upscale_deltas(target, source_patches=source)
        torch.testing.assert_close(actual, expected)
        assert not torch.allclose(actual, _upscale_deltas(target))
        source = target


def test_sampled_divisor_uses_previous_level_residuals() -> None:
    patches = torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4)
    patches[1] *= 1.5
    source = scale_patch_deviations(patches, SYNTHETIC_STD_SCALES[0])
    target = scale_patch_deviations(patches, SYNTHETIC_STD_SCALES[1])
    upscaled_source = F.interpolate(F.avg_pool2d(source, 2), scale_factor=2, mode="nearest")
    expected_residuals = (target - upscaled_source).flatten()
    stats = analysis._analyze_sample(target, 12, source_patches=source).calibration

    assert stats.residual_divisor_estimate == analysis._find_divisor(expected_residuals)
    assert stats.maximum_absolute_residual == expected_residuals.abs().max().item()

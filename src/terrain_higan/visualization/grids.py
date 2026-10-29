import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm

from terrain_higan.etopo.normalization import denormalize_elevation
from terrain_higan.visualization.terrain import (
    DEFAULT_MAX_ELEV_M,
    DEFAULT_MIN_ELEV_M,
    build_terrain_lut,
)


LUT_SIZE = 1024

_terrain_lut_cache: Optional[torch.Tensor] = None
_score_lut_cache: Optional[torch.Tensor] = None


def _move_lut_to_device(lut: torch.Tensor, device: Optional[torch.device]) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return lut.to(device)


def _get_terrain_lut(device: Optional[torch.device] = None) -> torch.Tensor:
    global _terrain_lut_cache

    if _terrain_lut_cache is None:
        _terrain_lut_cache = build_terrain_lut(
            LUT_SIZE,
            min_elev=DEFAULT_MIN_ELEV_M,
            max_elev=DEFAULT_MAX_ELEV_M,
        )
    if device is not None and _terrain_lut_cache.device != device:
        _terrain_lut_cache = _terrain_lut_cache.to(device)
    return _move_lut_to_device(_terrain_lut_cache, device)


def _colorize_with_lut(values: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    indices = (values * (LUT_SIZE - 1)).long()
    indices = torch.clamp(indices, 0, LUT_SIZE - 1)

    n_images, height, width = values.shape
    colors_hwc = lut[indices.reshape(-1)].reshape(n_images, height, width, 3)
    return colors_hwc.permute(0, 3, 1, 2)


def _near_square_grid_shape(n_images: int) -> tuple[int, int]:
    best_cols = 1
    best_rows = n_images
    best_diff = abs(best_rows - best_cols)
    best_area = best_rows * best_cols

    for candidate_cols in range(1, n_images + 1):
        candidate_rows = math.ceil(n_images / candidate_cols)
        diff = abs(candidate_rows - candidate_cols)
        area = candidate_rows * candidate_cols
        if diff < best_diff or (diff == best_diff and area < best_area):
            best_cols = candidate_cols
            best_rows = candidate_rows
            best_diff = diff
            best_area = area

    return best_rows, best_cols


def _make_grid(
    images: torch.Tensor,
    n_images: Optional[int],
    n_cols: Optional[int],
    upscale_factor: int,
    buffer_size: int,
    colorize: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if n_images is None:
        n_images = images.shape[0]
    images = images[:n_images]
    n_images_actual = images.shape[0]

    if n_images_actual == 0:
        raise ValueError("No images available to create a grid.")

    images = colorize(images)
    images = F.interpolate(images, scale_factor=upscale_factor, mode="nearest")

    channels, height, width = images.shape[1], images.shape[2], images.shape[3]
    if n_cols is None:
        n_rows, n_cols = _near_square_grid_shape(n_images_actual)
    else:
        n_rows = (n_images_actual + n_cols - 1) // n_cols

    if n_images_actual < n_rows * n_cols:
        padding_images = torch.zeros(
            n_rows * n_cols - n_images_actual,
            channels,
            height,
            width,
            dtype=images.dtype,
            device=images.device,
        )
        images = torch.cat([images, padding_images], dim=0)

    images_with_buffer = F.pad(images, (0, buffer_size, 0, buffer_size), mode="constant", value=0.0)
    buffer_height = images_with_buffer.shape[2]
    buffer_width = images_with_buffer.shape[3]

    grid = images_with_buffer.view(n_rows, n_cols, channels, buffer_height, buffer_width)
    grid = grid.permute(0, 2, 3, 1, 4).contiguous()
    grid = grid.view(n_rows, channels, buffer_height, n_cols * buffer_width)
    grid = grid.permute(1, 0, 2, 3).contiguous()
    return grid.view(channels, n_rows * buffer_height, n_cols * buffer_width)


def create_image_grid(
    images: torch.Tensor,
    n_images: Optional[int] = None,
    n_cols: Optional[int] = None,
    upscale_factor: int = 4,
    buffer_size: int = 4,
) -> torch.Tensor:
    """Create a grid from single-channel terrain images in normalized elevation units."""

    def colorize_terrain(batch: torch.Tensor) -> torch.Tensor:
        min_elev_meters = DEFAULT_MIN_ELEV_M
        max_elev_meters = DEFAULT_MAX_ELEV_M
        height_values_meters = denormalize_elevation(batch[:, 0, :, :])
        height_values_meters = torch.clamp(height_values_meters, min_elev_meters, max_elev_meters)
        normalized = (height_values_meters - min_elev_meters) / (max_elev_meters - min_elev_meters)
        return _colorize_with_lut(torch.clamp(normalized, 0.0, 1.0), _get_terrain_lut(batch.device))

    return _make_grid(images, n_images, n_cols, upscale_factor, buffer_size, colorize_terrain)


def make_logging_image_grid(
    images: torch.Tensor,
    max_images: int = 16,
    upscale_factor: int = 4,
) -> torch.Tensor:
    imgs = images.detach()
    return create_image_grid(imgs, n_images=min(imgs.shape[0], max_images), upscale_factor=upscale_factor)

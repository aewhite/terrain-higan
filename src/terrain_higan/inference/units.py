"""Elevation and horizontal-resolution conversions."""

import numpy as np
import torch

from .constants import ELEVATION_MEAN_M, ELEVATION_STD_M


def normalize_elevation(values: torch.Tensor | np.ndarray | float):
    return (values - ELEVATION_MEAN_M) / ELEVATION_STD_M


def denormalize_elevation(values: torch.Tensor | np.ndarray | float):
    return values * ELEVATION_STD_M + ELEVATION_MEAN_M


def meters_per_pixel(zoom: int) -> float:
    if isinstance(zoom, bool) or not isinstance(zoom, int) or not 0 <= zoom <= 20:
        raise ValueError("zoom must be an integer in 0..20")
    return 512.0 * 2.0 ** (10 - zoom)

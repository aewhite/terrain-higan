"""Colormaps local to the inference-only map-to-image demo."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps as matplotlib_colormaps

MIN_ELEVATION_M = -11000.0
MAX_ELEVATION_M = 9000.0
FROST_LINE_M = 5000.0

LAND_GAMMA = 0.7
LAND_TERRAIN_START = 0.25
LARGE_HILLSHADE_SIGMA = 6.0
SMALL_HILLSHADE_SIGMA = 1.2
LARGE_HILLSHADE_WEIGHT = 0.75
SMALL_HILLSHADE_WEIGHT = 0.25
HILLSHADE_GAMMA = 0.85
AMBIENT_LIGHT = 0.35
DIRECTIONAL_LIGHT = 0.65
SUN_AZIMUTH_DEGREES = 315.0
SUN_ALTITUDE_DEGREES = 45.0
REALISTIC_SHADOW_STRENGTH = 0.18
REALISTIC_SHADOW_MIN_INTENSITY = 0.88
OCEAN_REFERENCE_DEPTH_M = 10_000.0
OCEAN_COAST_RGB = np.array((0.68, 0.88, 1.0), dtype=np.float32)
OCEAN_DEEP_RGB = np.array((0.0, 0.10, 0.45), dtype=np.float32)
REALISTIC_OCEAN_HEIGHTS = (-11000.0, -100.0, -50.0, -25.0, -0.001)
REALISTIC_OCEAN_COLORS = ("082c50", "082c50", "176596", "8ed5e5", "8ed5e5")
REALISTIC_LAND_HEIGHTS = (
    0.0, 100.0, 700.0, 1400.0, 2000.0, 3000.0, 4000.0,
    FROST_LINE_M - 150.0, FROST_LINE_M, 9000.0,
)
REALISTIC_LAND_COLORS = (
    "c9bd7b", "6c9850", "547548", "958064", "858580",
    "9e927f", "b7ad9f", "d3d1cc", "f4f6f7", "f4f6f7",
)


def greyscale(elevation_m: np.ndarray) -> np.ndarray:
    finite = np.isfinite(elevation_m)
    normalized = _normalize(np.where(finite, elevation_m, MIN_ELEVATION_M))
    channel = (normalized * 255).astype(np.uint8)
    return np.repeat(channel[..., np.newaxis], 3, axis=-1)


def elevation(elevation_m: np.ndarray) -> np.ndarray:
    return render_elevation_relief(elevation_m)


def render_elevation_relief(elevation_m: np.ndarray) -> np.ndarray:
    """Render elevation with a hypsometric land palette and multi-scale relief."""
    elevation = np.asarray(elevation_m, dtype=np.float32)
    finite = np.isfinite(elevation)
    safe_elevation = np.where(finite, elevation, 0.0)
    land = safe_elevation >= 0.0
    rgb = np.empty((*elevation.shape, 3), dtype=np.float32)

    land_elevation = np.maximum(safe_elevation, 0.0)
    land_normalized = np.clip(land_elevation / MAX_ELEVATION_M, 0.0, 1.0)
    terrain_position = LAND_TERRAIN_START + np.power(land_normalized, LAND_GAMMA) * (1.0 - LAND_TERRAIN_START)
    rgb[land] = matplotlib_colormaps["terrain"](terrain_position[land])[..., :3]
    rgb[land] *= (AMBIENT_LIGHT + DIRECTIONAL_LIGHT * compute_hillshade(land_elevation)[land])[..., np.newaxis]

    ocean = ~land
    if np.any(ocean):
        depth = np.maximum(-safe_elevation[ocean], 0.0)
        ocean_position = np.power(np.clip(depth / OCEAN_REFERENCE_DEPTH_M, 0.0, 1.0), LAND_GAMMA)
        rgb[ocean] = OCEAN_COAST_RGB * (1.0 - ocean_position[..., np.newaxis]) + OCEAN_DEEP_RGB * ocean_position[..., np.newaxis]

    rgb[~finite] = 0.0
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def compute_hillshade(elevation_m: np.ndarray) -> np.ndarray:
    """Return northwest-lit, multi-scale hillshade values in the 0..1 range."""
    elevation = np.nan_to_num(np.asarray(elevation_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    large = _single_scale_hillshade(_gaussian_blur(elevation, LARGE_HILLSHADE_SIGMA))
    small = _single_scale_hillshade(_gaussian_blur(elevation, SMALL_HILLSHADE_SIGMA))
    return np.power(np.clip(LARGE_HILLSHADE_WEIGHT * large + SMALL_HILLSHADE_WEIGHT * small, 0.0, 1.0), HILLSHADE_GAMMA)


def compute_realistic_shadow(elevation_m: np.ndarray) -> np.ndarray:
    """Return broad hillshade for the subtle realistic-terrain shadow pass."""
    elevation = np.nan_to_num(np.asarray(elevation_m, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return _single_scale_hillshade(_gaussian_blur(elevation, LARGE_HILLSHADE_SIGMA))


def _single_scale_hillshade(elevation: np.ndarray) -> np.ndarray:
    gradient_y = np.zeros_like(elevation)
    gradient_x = np.zeros_like(elevation)
    if elevation.shape[1] > 1:
        gradient_x[:, 0] = elevation[:, 1] - elevation[:, 0]
        gradient_x[:, -1] = elevation[:, -1] - elevation[:, -2]
        gradient_x[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) * 0.5
    if elevation.shape[0] > 1:
        gradient_y[0] = elevation[1] - elevation[0]
        gradient_y[-1] = elevation[-1] - elevation[-2]
        gradient_y[1:-1] = (elevation[2:] - elevation[:-2]) * 0.5
    azimuth = np.deg2rad(SUN_AZIMUTH_DEGREES)
    altitude = np.deg2rad(SUN_ALTITUDE_DEGREES)
    light_x = np.sin(azimuth) * np.cos(altitude)
    # Array rows increase southward, so a northward light vector has negative Y.
    light_y = -np.cos(azimuth) * np.cos(altitude)
    light_z = np.sin(altitude)
    normal_scale = np.sqrt(gradient_x ** 2 + gradient_y ** 2 + 1.0)
    illumination = (-gradient_x * light_x - gradient_y * light_y + light_z) / normal_scale
    return np.clip((illumination + 1.0) * 0.5, 0.0, 1.0)


def _gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(sigma * 3.0)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded_x = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    horizontal = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded_x)
    padded_y = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="valid"), 0, padded_y)


def realistic(elevation_m: np.ndarray) -> np.ndarray:
    elevation_m = np.asarray(elevation_m, dtype=np.float32)
    finite = np.isfinite(elevation_m)
    elevation_m = np.where(finite, elevation_m, 0.0)
    rgb = np.empty((*elevation_m.shape, 3), dtype=np.float32)
    ocean = elevation_m < 0.0
    rgb[ocean] = _interpolate_colors(
        elevation_m[ocean],
        REALISTIC_OCEAN_HEIGHTS,
        REALISTIC_OCEAN_COLORS,
    )
    rgb[~ocean] = _interpolate_colors(
        elevation_m[~ocean],
        REALISTIC_LAND_HEIGHTS,
        REALISTIC_LAND_COLORS,
    )
    land_elevation = np.maximum(elevation_m, 0.0)
    flat_hillshade = (np.sin(np.deg2rad(SUN_ALTITUDE_DEGREES)) + 1.0) * 0.5
    shadow_delta = np.minimum(compute_realistic_shadow(land_elevation)[~ocean] - flat_hillshade, 0.0)
    intensity = np.clip(1.0 + REALISTIC_SHADOW_STRENGTH * shadow_delta, REALISTIC_SHADOW_MIN_INTENSITY, 1.0)
    rgb[~ocean] *= intensity[..., np.newaxis]
    rgb[~finite] = 0.0
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def render(name: str, elevation_m: np.ndarray) -> np.ndarray:
    try:
        renderer = {"greyscale": greyscale, "elevation": elevation, "realistic": realistic}[name]
    except KeyError as error:
        raise ValueError(f"Unknown colormap {name!r}") from error
    return renderer(elevation_m)


def _normalize(elevation_m: np.ndarray) -> np.ndarray:
    return np.clip((np.asarray(elevation_m) - MIN_ELEVATION_M) / (MAX_ELEVATION_M - MIN_ELEVATION_M), 0.0, 1.0)


def _interpolate_colors(values: np.ndarray, positions: tuple[float, ...], colors: tuple[str, ...]) -> np.ndarray:
    palette = np.array([tuple(int(color[index:index + 2], 16) / 255 for index in (0, 2, 4)) for color in colors])
    return np.stack([np.interp(values, positions, palette[:, channel]) for channel in range(3)], axis=-1)

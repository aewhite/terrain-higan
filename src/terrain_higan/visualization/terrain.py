from typing import Literal, Optional

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_MIN_ELEV_M = -11000.0
DEFAULT_MAX_ELEV_M = 8000.0
DEFAULT_ARRAY_MAX_ELEV_M = 9000.0


def build_tensorboard_terrain_colormap(
    min_elev: float = DEFAULT_MIN_ELEV_M,
    max_elev: float = DEFAULT_MAX_ELEV_M,
    name: str = "terrain_tensorboard",
) -> LinearSegmentedColormap:
    sea_level_norm = -min_elev / (max_elev - min_elev)
    epsilon = 0.0001

    position_color_pairs = [
        (0.0, (0.0, 0.0, 0.15, 1.0)),
        (sea_level_norm * 0.15, (0.0, 0.05, 0.25, 1.0)),
        (sea_level_norm * 0.3, (0.0, 0.1, 0.4, 1.0)),
        (sea_level_norm * 0.5, (0.05, 0.2, 0.6, 1.0)),
        (sea_level_norm * 0.7, (0.1, 0.35, 0.75, 1.0)),
        (sea_level_norm * 0.85, (0.15, 0.45, 0.85, 1.0)),
        (sea_level_norm - epsilon, (0.2, 0.5, 0.9, 1.0)),
        (sea_level_norm + epsilon, (0.2, 0.6, 0.2, 1.0)),
        (sea_level_norm + (1 - sea_level_norm) * 0.2, (0.6, 0.8, 0.3, 1.0)),
        (sea_level_norm + (1 - sea_level_norm) * 0.5, (0.7, 0.6, 0.4, 1.0)),
        (sea_level_norm + (1 - sea_level_norm) * 0.8, (0.5, 0.4, 0.3, 1.0)),
        (1.0, (0.9, 0.9, 0.9, 1.0)),
    ]

    position_color_pairs.sort(key=lambda x: x[0])
    position_color_pairs[0] = (0.0, position_color_pairs[0][1])
    position_color_pairs[-1] = (1.0, position_color_pairs[-1][1])
    return LinearSegmentedColormap.from_list(name, position_color_pairs)


def normalize_meters_for_colormap(
    elevation: np.ndarray,
    min_elev: float = DEFAULT_MIN_ELEV_M,
    max_elev: float = DEFAULT_MAX_ELEV_M,
) -> np.ndarray:
    """Apply terrain colormap scaling to raw elevation values in meters."""
    return np.clip((elevation - min_elev) / (max_elev - min_elev), 0.0, 1.0)


def apply_tensorboard_terrain_colormap_to_array(
    height_map: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    output_range: Literal["0-1", "0-255"] = "0-255",
) -> np.ndarray:
    """Apply the terrain colormap to a 2D numpy height map in meters."""
    if vmin is None:
        vmin = DEFAULT_MIN_ELEV_M
    if vmax is None:
        vmax = DEFAULT_ARRAY_MAX_ELEV_M

    terrain_cmap = build_tensorboard_terrain_colormap(
        min_elev=vmin,
        max_elev=vmax,
        name="terrain_tensorboard",
    )
    rgb_np = terrain_cmap(normalize_meters_for_colormap(height_map, vmin, vmax))[..., :3]

    if output_range == "0-255":
        return (rgb_np * 255).astype(np.uint8)
    return rgb_np.astype(np.float32)


def elevation_to_terrain_rgb(
    elevation_tensor: torch.Tensor,
    min_elev: float = DEFAULT_MIN_ELEV_M,
    max_elev: float = DEFAULT_MAX_ELEV_M,
) -> torch.Tensor:
    """Convert an elevation tensor in meters to CHW RGB values in [0, 1]."""
    device = elevation_tensor.device
    elev_np = elevation_tensor.detach().cpu().numpy()

    terrain_cmap = build_tensorboard_terrain_colormap(
        min_elev=min_elev,
        max_elev=max_elev,
        name="terrain_custom",
    )
    rgb_np = terrain_cmap(normalize_meters_for_colormap(elev_np, min_elev, max_elev))[..., :3]
    rgb_tensor = torch.from_numpy(rgb_np).float()

    if len(elevation_tensor.shape) == 2:
        rgb_tensor = rgb_tensor.permute(2, 0, 1)
    elif len(elevation_tensor.shape) == 3:
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)
    else:
        dims = list(range(rgb_tensor.ndim))
        rgb_idx = dims.pop(-1)
        dims.insert(-2, rgb_idx)
        rgb_tensor = rgb_tensor.permute(*dims)

    return rgb_tensor.to(device)


def build_terrain_lut(
    lut_size: int,
    min_elev: float = DEFAULT_MIN_ELEV_M,
    max_elev: float = DEFAULT_MAX_ELEV_M,
) -> torch.Tensor:
    terrain_cmap = build_tensorboard_terrain_colormap(
        min_elev=min_elev,
        max_elev=max_elev,
        name="terrain_lut",
    )
    rgb = terrain_cmap(np.linspace(0.0, 1.0, lut_size))[..., :3]
    return torch.from_numpy(rgb).float()

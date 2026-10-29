"""Inference-only model definitions compatible with current project checkpoints."""

import torch
import torch.nn as nn
import torch.nn.functional as functional

from terrain_higan.models import deep_conv2d
from terrain_higan.models.upscaling import (
    UPSCALER_LAYER_SPECS,
    mean_preserving_bilinear_2x,
)

from .constants import SIN_COS_FREQUENCIES, UPSCALE_SETTINGS, UPSCALER_SHRINK, Z0_OUTPUT_RANGE_M, residual_scale
from .units import normalize_elevation


class Z0Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._network = nn.Sequential(
            deep_conv2d(1, "128/5 128/5 128/5 64/3 16/3 4/1"),
            nn.Conv2d(4, 1, kernel_size=1),
        )
        self._output_min = float(normalize_elevation(Z0_OUTPUT_RANGE_M[0]))
        self._output_max = float(normalize_elevation(Z0_OUTPUT_RANGE_M[1]))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        noise = torch.clamp(noise, -5.0, 5.0)
        return torch.clamp(self._network(noise), self._output_min, self._output_max)


class Upscaler(nn.Module):
    """A map upscaler that composes residuals into elevation internally."""

    def __init__(
        self,
        *,
        checkpoint_zoom: int,
        layer_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if not 1 <= checkpoint_zoom <= 20:
            raise ValueError("checkpoint_zoom must be in 1..20")
        if len(layer_channels) != 4:
            raise ValueError("layer_channels must contain four entries")
        self._network = nn.Sequential(
            deep_conv2d(9, UPSCALER_LAYER_SPECS.format(*layer_channels)),
            nn.Conv2d(layer_channels[-1], 1, kernel_size=1),
        )
        self._checkpoint_zoom = checkpoint_zoom
        settings = UPSCALE_SETTINGS[checkpoint_zoom]
        self._output_min = settings.elevation_min
        self._output_max = settings.elevation_max
        self._residual_min = settings.residual_min
        self._residual_max = settings.residual_max

    def forward(self, input_map: torch.Tensor, noise: torch.Tensor, *, destination_zoom: int) -> torch.Tensor:
        if noise.shape[1] != 4:
            raise ValueError("noise must have four channels")
        noise = torch.clamp(noise, -5.0, 5.0)
        features = [input_map]
        for frequency in SIN_COS_FREQUENCIES:
            phase = frequency * 2.0 * torch.pi * input_map
            features.extend((torch.sin(phase), torch.cos(phase)))
        features.append(noise)
        normalized_residual = self._network(torch.cat(features, dim=1)).clamp(-5.0, 5.0)
        upscaled_input = functional.interpolate(input_map, scale_factor=2, mode="nearest")
        residual = torch.sinh(normalized_residual) * residual_scale(destination_zoom)
        halo = UPSCALER_SHRINK // 2
        parent = mean_preserving_bilinear_2x(input_map)[..., halo:-halo, halo:-halo]
        # Residual statistics measure every change relative to the nearest-upscaled parent.
        reference_parent = upscaled_input[
            ..., halo:-halo, halo:-halo
        ]
        residual = residual + (parent - reference_parent)
        parent = reference_parent
        residual = torch.clamp(residual, self._residual_min, self._residual_max)
        return torch.clamp(parent + residual, self._output_min, self._output_max)

"""Neural models for map-based z(n) terrain upscaling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from terrain_higan.etopo.normalization import (
    UPSCALE_STATS,
    apply_sin_cos_filters_at_frequencies,
    denormalize_residuals,
    normalize_downscaled_patches,
)
from terrain_higan.models import deep_conv2d
from terrain_higan.models.context_aware_conv2d import ContextAwareConv2d
from terrain_higan.models.responsive_clipper import ResponsiveClipper
from terrain_higan.models.upscaling import (
    UPSCALER_LAYER_SPECS,
    UPSCALER_SHRINK as _CONVOLUTION_SHRINK,
    mean_preserving_bilinear_2x,
)

_INPUT_CHANNELS = 9
_RESIDUAL_CHANNELS = 1
SIN_COS_FREQUENCIES = (8.0, 40.0)


def upscale_input_map_2x(input_map: torch.Tensor) -> torch.Tensor:
    return F.interpolate(input_map, scale_factor=2, mode="nearest")


class ZnUpscalerModel(nn.Module):
    def __init__(
            self,
            *,
            z_level: int,
            layer_channels: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(layer_channels) != 4:
            raise ValueError("layer_channels must contain four entries")
        self._network = nn.Sequential(
            deep_conv2d(_INPUT_CHANNELS, UPSCALER_LAYER_SPECS.format(*layer_channels)),
            nn.Conv2d(layer_channels[-1], _RESIDUAL_CHANNELS, kernel_size=1),
        )
        stats = UPSCALE_STATS[z_level]
        self._normalized_residual_clipper = ResponsiveClipper(-5.0, 5.0)
        self._residual_clipper = ResponsiveClipper(
            stats.dataset_minimum_raw_residual,
            stats.dataset_maximum_raw_residual,
        )
        self._output_clipper = ResponsiveClipper(
            stats.dataset_minimum_value,
            stats.dataset_maximum_value,
        )
        self._z_level = z_level

    def forward(
            self,
            input_map: torch.Tensor,
            noise: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.clamp(noise, -5.0, 5.0)
        sin_cos_filters = apply_sin_cos_filters_at_frequencies(input_map, SIN_COS_FREQUENCIES)
        normalized_residuals = self._network(torch.cat((input_map, sin_cos_filters, noise), dim=1))
        upscaled_input_map = upscale_input_map_2x(input_map)
        denormalized_residuals = denormalize_residuals(
            self._normalized_residual_clipper(normalized_residuals), self._z_level,
        )
        crop_size = _CONVOLUTION_SHRINK // 2
        base_input_map = mean_preserving_bilinear_2x(input_map)[
            ..., crop_size:-crop_size, crop_size:-crop_size
        ]
        # Residual statistics measure every change relative to the nearest-upscaled parent.
        reference_input_map = upscaled_input_map[
            ..., crop_size:-crop_size, crop_size:-crop_size
        ]
        denormalized_residuals = denormalized_residuals + (
            base_input_map - reference_input_map
        )
        denormalized_corrected_residuals = self._residual_clipper(denormalized_residuals)
        output_map = self._output_clipper(
            reference_input_map + denormalized_corrected_residuals
        )
        if self.training:
            return torch.cat((denormalized_residuals, output_map), dim=1)
        return output_map


class ZnUpscalerCritic(nn.Module):
    def __init__(self, *, z_level: int, layer_specs: str) -> None:
        super().__init__()
        parsed_layer_specs = layer_specs.split()
        first_layer_spec, *remaining_layer_specs = parsed_layer_specs
        first_channels, first_kernel_size = (
            int(value) for value in first_layer_spec.split("/")
        )
        final_channels = int(parsed_layer_specs[-1].split("/")[0])
        self._first_layer = ContextAwareConv2d(
            1,
            first_channels,
            context_channels=1,
            kernel_size=first_kernel_size,
        )
        self._first_layer.conv = spectral_norm(self._first_layer.conv)
        layers: list[nn.Module] = [nn.LeakyReLU()]
        if remaining_layer_specs:
            layers.append(
                deep_conv2d(
                    first_channels,
                    " ".join(remaining_layer_specs),
                    activation=nn.LeakyReLU,
                    spectral_norm=False,
                )
            )
        layers.append(nn.Conv2d(final_channels, 1, kernel_size=1))
        self._network = nn.Sequential(*layers)
        self._output_clipper = ResponsiveClipper(-10.0, 10.0)
        self._z_level = z_level

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        normalized_patches, patch_means = normalize_downscaled_patches(
            patches,
            self._z_level,
        )
        features = self._first_layer(normalized_patches, patch_means.flatten(start_dim=1))
        return self._output_clipper(self._network(features).flatten(start_dim=1))

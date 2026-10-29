import math
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from terrain_higan.etopo.analyze_patches import evaluate_patch_mean_pdf
from terrain_higan.etopo.normalization import (
    ELEVATION_STD_M,
    SYNTHETIC_SOURCE_Z_LEVEL,
    SYNTHETIC_STD_SCALES,
    UPSCALE_STATS,
    apply_sin_cos_filters_at_frequencies,
    denormalize_elevation,
    normalize_downscaled_patches,
    normalize_residuals,
    scale_patch_deviations,
)
from terrain_higan.projects.z0_generator.config import GENERATOR_SHRINK
from terrain_higan.projects.z0_generator.export import EXPORT_MINIMUM_INPUT_SIZE
from terrain_higan.projects.zn_map_upscaler.config import (
    LARGE_MAP_PREVIEW_TARGET_SIZE,
    METRIC_CONTEXT_MARGIN,
    Z0_MINIMUM_NOISE_SIZE,
)
from terrain_higan.projects.zn_map_upscaler.critic_previews import make_scored_patch_grid
from terrain_higan.projects.zn_map_upscaler.geometry import center_crop
from terrain_higan.projects.zn_map_upscaler.models import (
    SIN_COS_FREQUENCIES,
    _CONVOLUTION_SHRINK,
    ZnUpscalerCritic,
    ZnUpscalerModel,
)
from terrain_higan.projects.zn_map_upscaler.save import load_zn_map_upscaler
from terrain_higan.projects.zn_map_upscaler.profiling import profile_range
from terrain_higan.training import TrainingContext
from terrain_higan.training.losses import residual_aggregate_loss
from terrain_higan.visualization import make_logging_image_grid
from terrain_higan.visualization.terrain import DEFAULT_MAX_ELEV_M, DEFAULT_MIN_ELEV_M

NORMALIZED_PATCH_LIMIT = 5.0


@dataclass
class ZnUpscalerStepData:
    """Maps/patches use universal normalized elevations; residuals are differences,
    not meters. Training publishes detached diagnostics after both updates.
    """

    real_patches: torch.Tensor = field(init=False)
    real_source_patches: torch.Tensor = field(init=False)
    input_map: torch.Tensor = field(init=False)
    denormalized_residuals: torch.Tensor = field(init=False)
    denormalized_residual_patches: torch.Tensor = field(init=False)
    output_map: torch.Tensor = field(init=False)
    generated_patches: torch.Tensor = field(init=False)
    real_scores: torch.Tensor = field(init=False)
    fake_scores: torch.Tensor = field(init=False)
    generator_adversarial_loss: torch.Tensor = field(init=False)
    generator_residual_aggregate_loss: torch.Tensor = field(init=False)
    generator_residual_bounds_loss: torch.Tensor = field(init=False)
    generator_residual_block_mean_loss: torch.Tensor = field(init=False)
    generator_residual_block_mean_loss_weighted: torch.Tensor = field(init=False)
    real_residual_block_means: torch.Tensor = field(init=False)
    generator_loss: torch.Tensor = field(init=False)
    critic_loss: torch.Tensor = field(init=False)
    input_metric_patches: torch.Tensor = field(init=False)
    output_metric_rows: torch.Tensor = field(init=False)
    output_metric_columns: torch.Tensor = field(init=False)


class ScaleRealPatchesStage:
    """Build current-level targets and previous-level sources from normalized z10 patches."""

    def __init__(self, z_level: int) -> None:
        index = z_level - SYNTHETIC_SOURCE_Z_LEVEL - 1
        self._scale = SYNTHETIC_STD_SCALES[index]
        self._source_scale = 1.0 if index == 0 else SYNTHETIC_STD_SCALES[index - 1]

    def run(self, context: TrainingContext[ZnUpscalerStepData]) -> None:
        z10_patches = context.step_data.real_patches
        context.step_data.real_source_patches = (
            z10_patches if self._source_scale == 1.0
            else scale_patch_deviations(z10_patches, self._source_scale)
        )
        context.step_data.real_patches = scale_patch_deviations(
            z10_patches, self._scale
        )


class MaterializedInputMapStage:
    """Select balanced metric crops and their convolution context from z(n-1) maps."""

    def __init__(
            self,
            *,
            maps_dir: Path,
            z_level: int,
            source_level_name: str | None = None,
            maps_per_batch: int,
            metric_patch_size: int,
            input_map_patch_mean_pdf: torch.Tensor,
            selected_patch_count: int,
            inverse_pdf_weight_central_fraction: float,
            inverse_pdf_weight_cap_stddevs: float,
            device: torch.device,
            loading_workers: int = 2,
    ) -> None:
        source_dir = maps_dir / (source_level_name or f"z{z_level - 1}")
        self._map_paths = tuple(sorted(source_dir.glob("maps_*.npy")))
        if not self._map_paths:
            raise FileNotFoundError(f"No materialized map batches found in {source_dir}")
        if selected_patch_count > maps_per_batch:
            raise ValueError(
                f"selected_patch_count {selected_patch_count} exceeds maps_per_batch {maps_per_batch}"
            )
        self._maps_per_batch = maps_per_batch
        self._metric_patch_size = metric_patch_size
        # Bilinear interpolation needs a neighboring parent cell beyond the convolution halo.
        self._context_margin = METRIC_CONTEXT_MARGIN + 1
        self._input_map_patch_mean_pdf = input_map_patch_mean_pdf
        self._selected_patch_count = selected_patch_count
        self._central_fraction = inverse_pdf_weight_central_fraction
        self._cap_stddevs = inverse_pdf_weight_cap_stddevs
        self._device = device
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="input-maps")
            if self._stream is not None else None
        )
        self._staging_executor = (
            ThreadPoolExecutor(max_workers=loading_workers, thread_name_prefix="input-map-staging")
            if self._stream is not None else None
        )
        self._pinned_buffers = [[], []]
        self._transfer_events = [None, None]
        self._next_buffer_index = 0
        self._prefetched_batch = None
        self._file_index = 0
        self._map_offset = 0
        self._current_maps = None
        self._valid_rows = None
        self._valid_columns = None

    def run(self, context: TrainingContext[ZnUpscalerStepData]) -> None:
        with profile_range("input_maps/load_to_device"):
            maps, valid_rows, valid_columns = self._load_map_batch()
        with profile_range("input_maps/sample_metric_crops"):
            rows, columns = self._sample_metric_coordinates(valid_rows, valid_columns)
            map_indices = torch.arange(self._maps_per_batch, device=self._device)
            metric_patches = _extract_patches(
                maps, map_indices, rows, columns, self._metric_patch_size,
            )
        with profile_range("input_maps/metric_means"):
            metric_means = metric_patches.mean(dim=(1, 2, 3))
        with profile_range("input_maps/pdf_selection"):
            selected_indices = select_pdf_weighted_values(
                metric_means,
                self._input_map_patch_mean_pdf,
                sample_count=self._selected_patch_count,
                central_fraction=self._central_fraction,
                cap_stddevs=self._cap_stddevs,
            )
        with profile_range("input_maps/gather_context"):
            context.step_data.input_map = _extract_patches(
                maps,
                selected_indices,
                rows[selected_indices] - self._context_margin,
                columns[selected_indices] - self._context_margin,
                self._metric_patch_size + 2 * self._context_margin,
            )
            context.step_data.input_metric_patches = metric_patches[selected_indices]
            context.step_data.output_metric_rows = torch.zeros(
                self._selected_patch_count, dtype=torch.long, device=self._device,
            )
            context.step_data.output_metric_columns = torch.zeros(
                self._selected_patch_count, dtype=torch.long, device=self._device,
            )

    def _load_map_batch(self) -> tuple[torch.Tensor, int, int]:
        if self._stream is None:
            return self._read_map_batch()
        if self._prefetched_batch is None:
            self._prefetch_next()
        with profile_range("input_maps/wait_for_prefetch"):
            maps, valid_rows, valid_columns, ready = self._prefetched_batch.result()
        self._prefetched_batch = None
        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_event(ready)
        maps.record_stream(current_stream)
        self._prefetch_next()
        return maps, valid_rows, valid_columns

    def _prefetch_next(self) -> None:
        buffer_index = self._next_buffer_index
        self._next_buffer_index = 1 - buffer_index
        self._prefetched_batch = self._executor.submit(
            copy_context().run, self._prefetch_map_batch, buffer_index,
        )

    def _prefetch_map_batch(self, buffer_index: int):
        previous_transfer = self._transfer_events[buffer_index]
        if previous_transfer is not None:
            previous_transfer.synchronize()
        with torch.cuda.stream(self._stream):
            maps, valid_rows, valid_columns = self._read_map_batch(
                self._pinned_buffers[buffer_index],
            )
            ready = torch.cuda.Event()
            ready.record(self._stream)
        self._transfer_events[buffer_index] = ready
        return maps, valid_rows, valid_columns, ready

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._staging_executor.shutdown(wait=True, cancel_futures=True)
            self._staging_executor = None
            self._stream.synchronize()
            self._prefetched_batch = None
            self._pinned_buffers = [[], []]
        self._current_maps = None

    def _read_map_batch(
            self, pinned_buffers: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        remaining = self._maps_per_batch
        parts = []
        pending_files = []
        if self._current_maps is not None:
            count = min(remaining, self._current_maps.shape[0] - self._map_offset)
            parts.append(self._current_maps[self._map_offset:self._map_offset + count])
            remaining -= count
            self._map_offset += count
            if self._map_offset == self._current_maps.shape[0]:
                self._current_maps = None
                self._map_offset = 0
                self._file_index = (self._file_index + 1) % len(self._map_paths)
        while remaining:
            path = self._map_paths[self._file_index]
            maps = np.load(path, mmap_mode="c")
            self._valid_rows, self._valid_columns = self._validate_maps(path, maps)
            cpu_maps = torch.from_numpy(maps)
            staged = None
            if pinned_buffers is not None:
                buffer_index = len(pending_files)
                if buffer_index == len(pinned_buffers):
                    pinned_buffers.append(torch.empty_like(cpu_maps, pin_memory=True))
                elif pinned_buffers[buffer_index].shape != cpu_maps.shape:
                    pinned_buffers[buffer_index] = torch.empty_like(cpu_maps, pin_memory=True)
                staged = self._staging_executor.submit(
                    copy_context().run, self._stage_file, cpu_maps, pinned_buffers[buffer_index],
                )
                cpu_maps = pinned_buffers[buffer_index]
            count = min(remaining, maps.shape[0])
            pending_files.append((cpu_maps, staged, count))
            remaining -= count
            if count == maps.shape[0]:
                self._file_index = (self._file_index + 1) % len(self._map_paths)
            else:
                self._map_offset = count
        for cpu_maps, staged, count in pending_files:
            if staged is not None:
                staged.result()
            with profile_range("input_maps/file_to_device"):
                maps = cpu_maps.to(self._device, non_blocking=True)
            parts.append(maps[:count])
            if count < maps.shape[0]:
                self._current_maps = maps
        maps = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return maps, self._valid_rows, self._valid_columns

    @staticmethod
    def _stage_file(cpu_maps: torch.Tensor, pinned_maps: torch.Tensor) -> None:
        with profile_range("input_maps/stage_pinned"):
            pinned_maps.copy_(cpu_maps)

    def _validate_maps(self, path: Path, maps: np.ndarray) -> tuple[int, int]:
        if maps.ndim != 4 or maps.shape[1] != 1:
            raise ValueError(f"Expected B1HW maps in {path}, got {maps.shape}")
        if maps.shape[0] == 0:
            raise ValueError(f"No maps in {path}")
        valid_rows = maps.shape[-2] - self._metric_patch_size - 2 * self._context_margin + 1
        valid_columns = maps.shape[-1] - self._metric_patch_size - 2 * self._context_margin + 1
        if valid_rows <= 0 or valid_columns <= 0:
            raise ValueError(
                f"Cannot sample {self._metric_patch_size}x{self._metric_patch_size} metric patches "
                f"with a {self._context_margin}-cell context margin from {path} "
                f"with shape {maps.shape}"
            )

        return valid_rows, valid_columns

    def _sample_metric_coordinates(
            self, valid_rows: int, valid_columns: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.randint(valid_rows, (self._maps_per_batch,), device=self._device)
        columns = torch.randint(valid_columns, (self._maps_per_batch,), device=self._device)
        return rows + self._context_margin, columns + self._context_margin


class ZnUpscalerLargeMapPreviewStage:
    """Log a frozen z0-to-zn inference chain at approximately 2048 pixels."""

    def __init__(
            self,
            writer: SummaryWriter,
            *,
            z_level: int,
            model: ZnUpscalerModel,
            z0_pt2_path: Path,
            predecessor_model_paths: tuple[Path, ...],
            image_every_steps: int,
            device: torch.device,
            target_size: int = LARGE_MAP_PREVIEW_TARGET_SIZE,
            z0_minimum_noise_size: int = Z0_MINIMUM_NOISE_SIZE,
    ) -> None:
        self._writer = writer
        self._z_level = z_level
        self._model = model
        self._z0_model = torch.export.load(z0_pt2_path).module().to(device)
        self._predecessor_models = tuple(
            load_zn_map_upscaler(path).to(device).eval()
            for path in predecessor_model_paths
        )
        self._image_every_steps = image_every_steps
        self._device = device
        self._target_size = target_size
        self._z0_noise_size = calculate_large_map_z0_noise_size(
            z_level,
            target_size,
            minimum_noise_size=z0_minimum_noise_size,
        )

    def run(self, context: TrainingContext[ZnUpscalerStepData]) -> None:
        if context.step_number % self._image_every_steps != 0:
            return
        was_training = self._model.training
        self._model.eval()
        try:
            with profile_range("preview/inference"), torch.inference_mode():
                large_map = render_preview_tiled(
                    self._z0_model,
                    torch.randn(1, 1, self._z0_noise_size, self._z0_noise_size, device=self._device)
                )
                upscalers = (*self._predecessor_models, self._model)
                for upscaler_index, upscaler in enumerate(upscalers, start=1):
                    remaining_upscalers = self._z_level - upscaler_index + 1
                    large_map = center_crop(
                        large_map,
                        calculate_large_map_input_size(
                            remaining_upscalers,
                            self._target_size,
                        ),
                    )
                    large_map = render_preview_tiled(
                        upscaler,
                        large_map,
                        torch.randn(
                            1,
                            4,
                            large_map.shape[-2],
                            large_map.shape[-1],
                            device=self._device,
                            dtype=large_map.dtype,
                        ),
                    )
                large_map = center_crop(large_map, self._target_size)
        finally:
            self._model.train(was_training)
        with profile_range("preview/image_write"):
            self._writer.add_image(
                "images/large_scale_output_map",
                make_logging_image_grid(large_map, max_images=1, upscale_factor=1).cpu(),
                context.step_number,
            )


@torch.inference_mode()
def render_preview_tiled(
        model: torch.nn.Module,
        input_map: torch.Tensor,
        noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stitch 64x64 output tiles using the same input and noise convolution context."""
    scale = 1 if noise is None else 2
    shrink = GENERATOR_SHRINK if noise is None else _CONVOLUTION_SHRINK
    minimum_input_size = EXPORT_MINIMUM_INPUT_SIZE if noise is None else shrink // scale + 1
    height, width = (scale * size - shrink for size in input_map.shape[-2:])
    output = input_map.new_empty((input_map.shape[0], 1, height, width))
    for row in range(0, height, 64):
        row_end = min(row + 64, height)
        input_row_end = (row_end + shrink) // scale
        input_row_start = min(row // scale, input_row_end - minimum_input_size)
        input_rows = slice(input_row_start, input_row_end)
        tile_row_start = row - scale * input_row_start
        for column in range(0, width, 64):
            column_end = min(column + 64, width)
            input_column_end = (column_end + shrink) // scale
            input_column_start = min(column // scale, input_column_end - minimum_input_size)
            input_columns = slice(input_column_start, input_column_end)
            tile_column_start = column - scale * input_column_start
            tile_input = input_map[..., input_rows, input_columns]
            tile = (
                model(tile_input) if noise is None
                else model(tile_input, noise[..., input_rows, input_columns])
            )
            output[..., row:row_end, column:column_end] = tile[
                ...,
                tile_row_start:tile_row_start + row_end - row,
                tile_column_start:tile_column_start + column_end - column,
            ]
    return output


def residual_bounds_loss(
        denormalized_residuals: torch.Tensor,
        residual_limits: tuple[float, float],
        residual_divisor: float,
) -> torch.Tensor:
    """Average squared bound violations in zoom-normalized residual units."""
    minimum, maximum = residual_limits
    lower_violation = F.relu(minimum - denormalized_residuals) / residual_divisor
    upper_violation = F.relu(denormalized_residuals - maximum) / residual_divisor
    return (lower_violation.square() + upper_violation.square()).mean()


def residual_block_mean_loss(
        denormalized_residuals: torch.Tensor,
        residual_divisor: float,
) -> torch.Tensor:
    """Penalize each map's mean linear residual, scaled for this zoom.

    The legacy block_mean name is retained for config and logging compatibility.
    """
    patch_means = denormalized_residuals.mean(dim=(2, 3))
    return (patch_means / residual_divisor).square().mean()


class ZnUpscalerLossTrendStage:
    """After minimum_steps, continue while either full-window loss mean improves."""

    def __init__(self, lookback_steps: int = 10_000, *, minimum_steps: int = 50_000) -> None:
        if lookback_steps <= 0:
            raise ValueError("lookback_steps must be positive")
        if minimum_steps < 0:
            raise ValueError("minimum_steps must be non-negative")
        self._lookback_steps = lookback_steps
        self._minimum_steps = minimum_steps
        self._window_steps = 0
        self._critic_sum = None
        self._generator_sum = None
        self._previous_means = None
        self._should_continue = True

    @torch.no_grad()
    def run(self, context: TrainingContext[ZnUpscalerStepData]) -> None:
        step_data = context.step_data
        if self._critic_sum is None:
            self._critic_sum = step_data.critic_loss.new_zeros((), dtype=torch.float64)
            self._generator_sum = step_data.generator_adversarial_loss.new_zeros(
                (), dtype=torch.float64,
            )
        self._critic_sum.add_(step_data.critic_loss.detach())
        self._generator_sum.add_(step_data.generator_adversarial_loss.detach())
        self._window_steps += 1
        if self._window_steps < self._lookback_steps:
            return

        critic_mean, generator_mean = (
            torch.stack((self._critic_sum, self._generator_sum)) / self._window_steps
        ).tolist()
        if self._previous_means is not None and context.step_number >= self._minimum_steps:
            previous_critic_mean, previous_generator_mean = self._previous_means
            self._should_continue = (
                critic_mean > previous_critic_mean
                or generator_mean < previous_generator_mean
            )
        self._previous_means = critic_mean, generator_mean
        self._critic_sum.zero_()
        self._generator_sum.zero_()
        self._window_steps = 0
        print(
            f"Step {context.step_number:,}: {self._lookback_steps:,}-step loss means: "
            f"critic={critic_mean:.6g}, generator_adversarial={generator_mean:.6g}"
        )
        if not self._should_continue:
            print("Stopping training: neither windowed loss is improving")

    def should_continue(self, context: TrainingContext[ZnUpscalerStepData]) -> bool:
        return self._should_continue


class ZnUpscalerTrainStage:
    def __init__(
            self,
            *,
            model: ZnUpscalerModel,
            critic: ZnUpscalerCritic,
            z_level: int,
            model_optimizer: torch.optim.Optimizer,
            critic_optimizer: torch.optim.Optimizer,
            patch_mean_pdf: torch.Tensor,
            patch_size: int,
            patches_per_batch: int,
            residual_aggregate_loss_weight: float,
            residual_bounds_loss_weight: float,
            residual_block_mean_loss_weight: float,
            residual_limits: tuple[float, float],
            inverse_pdf_weight_central_fraction: float,
            inverse_pdf_weight_cap_stddevs: float,
    ) -> None:
        self._model = model
        self._critic = critic
        self._model_parameters = tuple(model.parameters())
        self._critic_parameters = tuple(critic.parameters())
        self._z_level = z_level
        self._model_optimizer = model_optimizer
        self._critic_optimizer = critic_optimizer
        self._patch_mean_pdf = patch_mean_pdf
        self._patch_size = patch_size
        self._patches_per_batch = patches_per_batch
        self._residual_aggregate_loss_weight = residual_aggregate_loss_weight
        self._residual_bounds_loss_weight = residual_bounds_loss_weight
        self._residual_block_mean_loss_weight = residual_block_mean_loss_weight
        self._residual_divisor = UPSCALE_STATS[z_level].residual_divisor
        self._residual_minimum, self._residual_maximum = residual_limits
        self._inverse_pdf_weight_central_fraction = inverse_pdf_weight_central_fraction
        self._inverse_pdf_weight_cap_stddevs = inverse_pdf_weight_cap_stddevs

    def run(self, context: TrainingContext[ZnUpscalerStepData]) -> None:
        step_data = context.step_data
        input_map = step_data.input_map
        with profile_range("train/noise"):
            noise = torch.randn(
                input_map.shape[0],
                4,
                input_map.shape[2],
                input_map.shape[3],
                dtype=input_map.dtype,
                device=input_map.device,
            )
        with profile_range("train/generator_forward"):
            output_maps = self._model(input_map, noise)
            output_region_size = 2 * step_data.input_metric_patches.shape[-1]
        with profile_range("train/generated_patch_extraction"):
            row_offset = (output_maps.shape[-2] - output_region_size) // 2
            column_offset = (output_maps.shape[-1] - output_region_size) // 2
            output_patches = sample_patches_from_regions(
                output_maps,
                rows=step_data.output_metric_rows + row_offset,
                columns=step_data.output_metric_columns + column_offset,
                region_size=output_region_size,
                patch_size=self._patch_size,
            )
            residual_patches, generated_patches = output_patches.split(1, dim=1)
        with profile_range("train/generated_residual_losses"):
            generated_normalized_residuals = normalize_residuals(
                residual_patches,
                self._z_level,
            )
            generator_residual_bounds_loss = residual_bounds_loss(
                residual_patches,
                (self._residual_minimum, self._residual_maximum),
                self._residual_divisor,
            )
            generator_residual_block_mean_loss = residual_block_mean_loss(
                output_maps[:, :1], self._residual_divisor,
            )
            generator_residual_block_mean_loss_weighted = (
                self._residual_block_mean_loss_weight * generator_residual_block_mean_loss
            )
        with profile_range("train/real_pdf_selection"):
            candidate_real_patches = step_data.real_patches
            real_indices = select_pdf_weighted_patches(
                candidate_real_patches,
                self._patch_mean_pdf,
                sample_count=self._patches_per_batch,
                central_fraction=self._inverse_pdf_weight_central_fraction,
                cap_stddevs=self._inverse_pdf_weight_cap_stddevs,
            )
            real_patches = candidate_real_patches[real_indices]
        with profile_range("train/real_residual_preparation"):
            source_real_patches = (
                step_data.real_source_patches[real_indices]
                if self._z_level > SYNTHETIC_SOURCE_Z_LEVEL else real_patches
            )
            real_downsampled_patches = F.avg_pool2d(source_real_patches, kernel_size=2)
            real_residuals = real_patches - F.interpolate(
                real_downsampled_patches, scale_factor=2, mode="nearest",
            )
            real_normalized_residuals = normalize_residuals(real_residuals, self._z_level)
            if self._z_level > SYNTHETIC_SOURCE_Z_LEVEL:
                step_data.real_residual_block_means = F.avg_pool2d(real_residuals, 2, 2).detach()
        with profile_range("train/critic_forward"):
            joint_patches = torch.cat((real_patches, generated_patches), dim=0)
            real_scores, fake_scores = torch.split(
                self._critic(joint_patches),
                (real_patches.shape[0], generated_patches.shape[0]),
            )

        with profile_range("train/aggregate_and_adversarial_losses"):
            generator_adversarial_loss = F.elu(-fake_scores).mean()
            generator_residual_aggregate_loss = residual_aggregate_loss(
                generated_normalized_residuals,
                real_normalized_residuals,
            )
            generator_loss = (
                    generator_adversarial_loss
                    + self._residual_aggregate_loss_weight * generator_residual_aggregate_loss
                    + self._residual_bounds_loss_weight * generator_residual_bounds_loss
                    + generator_residual_block_mean_loss_weighted
            )
            generated_critic_loss = F.elu(fake_scores).mean()
            real_critic_loss = F.elu(-real_scores).mean()
            critic_loss = generated_critic_loss + real_critic_loss

        # Both gradients come from the same forward pass, before either update.
        with profile_range("train/generator_gradients"):
            model_gradients = torch.autograd.grad(
                generator_loss, self._model_parameters, retain_graph=True,
            )
        with profile_range("train/critic_gradients"):
            critic_gradients = torch.autograd.grad(critic_loss, self._critic_parameters)
        with profile_range("train/assign_gradients"):
            _assign_gradients(self._model_parameters, model_gradients)
            _assign_gradients(self._critic_parameters, critic_gradients)
        with profile_range("train/generator_optimizer"):
            self._model_optimizer.step()
            self._model_optimizer.zero_grad(set_to_none=True)
        with profile_range("train/critic_optimizer"):
            self._critic_optimizer.step()
            self._critic_optimizer.zero_grad(set_to_none=True)

        logging_maps = center_crop(output_maps.detach(), output_region_size)
        denormalized_residuals, output_map = logging_maps.split(1, dim=1)
        step_data.denormalized_residuals = denormalized_residuals
        step_data.denormalized_residual_patches = residual_patches.detach()
        step_data.output_map = output_map
        step_data.real_patches = real_patches.detach()
        step_data.generated_patches = generated_patches.detach()
        step_data.real_scores = real_scores.detach()
        step_data.fake_scores = fake_scores.detach()
        step_data.generator_adversarial_loss = generator_adversarial_loss.detach()
        step_data.generator_residual_aggregate_loss = generator_residual_aggregate_loss.detach()
        step_data.generator_residual_bounds_loss = generator_residual_bounds_loss.detach()
        step_data.generator_residual_block_mean_loss = generator_residual_block_mean_loss.detach()
        step_data.generator_residual_block_mean_loss_weighted = generator_residual_block_mean_loss_weighted.detach()
        step_data.generator_loss = generator_loss.detach()
        step_data.critic_loss = critic_loss.detach()


class ZnUpscalerTensorBoardStage:
    def __init__(
            self,
            writer: SummaryWriter,
            *,
            z_level: int,
            log_every_steps: int,
            image_every_steps: int,
            critic_extreme_patch_count: int = 16,
    ) -> None:
        self._writer = writer
        self._z_level = z_level
        self._log_every_steps = log_every_steps
        self._image_every_steps = image_every_steps
        self._critic_extreme_patch_count = critic_extreme_patch_count

    def run(self, context: TrainingContext[ZnUpscalerStepData]) -> None:
        if context.step_number % self._log_every_steps == 0:
            with profile_range("logging/scalars"):
                self._log_scalars(context.step_data, context.step_number)
            with profile_range("logging/histograms"):
                self._log_histograms(context.step_data, context.step_number)

        if context.step_number % self._image_every_steps == 0:
            with profile_range("logging/images"):
                self._log_images(context.step_data, context.step_number)

    @torch.no_grad()
    def _log_scalars(self, step_data: ZnUpscalerStepData, step_number: int) -> None:
        losses = (
            ("loss/generator", step_data.generator_loss),
            ("loss/generator_adversarial", step_data.generator_adversarial_loss),
            ("loss/generator_residual_aggregate", step_data.generator_residual_aggregate_loss),
            ("loss/generator_residual_bounds", step_data.generator_residual_bounds_loss),
            ("loss/generator_residual_block_mean", step_data.generator_residual_block_mean_loss),
            ("loss/generator_residual_block_mean_weighted", step_data.generator_residual_block_mean_loss_weighted),
            ("loss/critic", step_data.critic_loss),
        )
        for tag, loss in losses:
            self._writer.add_scalar(tag, loss.item(), step_number)

        block_means_meters = F.avg_pool2d(step_data.denormalized_residuals, 2, 2) * ELEVATION_STD_M
        parent_height, parent_width = (size // 2 for size in step_data.output_map.shape[-2:])
        row_start = (step_data.input_metric_patches.shape[-2] - parent_height) // 2
        column_start = (step_data.input_metric_patches.shape[-1] - parent_width) // 2
        parent_map = step_data.input_metric_patches[
            ..., row_start:row_start + parent_height, column_start:column_start + parent_width
        ]
        output_parent_drift_meters = (
            F.avg_pool2d(step_data.output_map, 2, 2) - parent_map
        ) * ELEVATION_STD_M
        diagnostics = (
            ("residuals/block_mean_meters_mean", block_means_meters.mean()),
            ("residuals/block_mean_meters_rms", block_means_meters.square().mean().sqrt()),
            ("consistency/output_parent_drift_meters_mean", output_parent_drift_meters.mean()),
            ("consistency/output_parent_drift_meters_rms", output_parent_drift_meters.square().mean().sqrt()),
        )
        for tag, value in diagnostics:
            self._writer.add_scalar(tag, value.item(), step_number)
        if self._z_level > SYNTHETIC_SOURCE_Z_LEVEL:
            real_scaled_block_means = (
                step_data.real_residual_block_means / UPSCALE_STATS[self._z_level].residual_divisor
            )
            self._writer.add_scalar(
                "residuals/real_block_mean_scaled_rms",
                real_scaled_block_means.square().mean().sqrt().item(),
                step_number,
            )

    def _log_histograms(self, step_data: ZnUpscalerStepData, step_number: int) -> None:
        histograms = (
            ("residuals/denormalized", step_data.denormalized_residuals),
            ("scores/critic_real", step_data.real_scores),
            ("scores/critic_fake", step_data.fake_scores),
        )
        for tag, values in histograms:
            self._writer.add_histogram(tag, values, step_number)

    def _log_images(self, step_data: ZnUpscalerStepData, step_number: int) -> None:
        self._log_critic_extremes(step_data, step_number)
        terrain_images = (
            ("images/input_maps", step_data.input_map, 16, 4),
            ("images/input_metric_patches", step_data.input_metric_patches, 16, 16),
            ("images/output_maps", step_data.output_map, 16, 6),
            ("images/generated_patches", step_data.generated_patches, 64, 24),
        )
        for tag, maps, max_images, upscale_factor in terrain_images:
            self._writer.add_image(
                tag,
                make_logging_image_grid(
                    maps, max_images=max_images, upscale_factor=upscale_factor,
                ).cpu(),
                step_number,
            )

        output_map_high_pass, output_map_low_pass = calculate_sincos_filters(step_data.output_map[:16])
        output_features = (
            ("images/output_map_high_pass_features", output_map_high_pass),
            ("images/output_map_low_pass_features", output_map_low_pass),
        )
        for tag, features in output_features:
            self._writer.add_image(
                tag,
                _make_logging_rgb_image_grid(features, max_images=16, upscale_factor=2).cpu(),
                step_number,
            )

        input_map_high_pass, input_map_low_pass = calculate_sincos_filters(step_data.input_map[:16])
        input_features = (
            ("images/input_map_high_pass_features", input_map_high_pass),
            ("images/input_map_low_pass_features", input_map_low_pass),
        )
        for tag, features in input_features:
            self._writer.add_image(
                tag,
                _make_logging_rgb_image_grid(features, max_images=16, upscale_factor=4).cpu(),
                step_number,
            )

        self._writer.add_image(
            "images/real_patches",
            make_logging_image_grid(step_data.real_patches, max_images=64, upscale_factor=24).cpu(),
            step_number,
        )
        normalized_images = (
            ("images/real_patches_normalized", step_data.real_patches),
            ("images/generated_patches_normalized", step_data.generated_patches),
        )
        for tag, patches in normalized_images:
            self._writer.add_image(
                tag,
                _make_normalized_patch_logging_image_grid(
                    patches, self._z_level, max_images=64, upscale_factor=24,
                ).cpu(),
                step_number,
            )

    @torch.no_grad()
    def _log_critic_extremes(self, step_data: ZnUpscalerStepData, step_number: int) -> None:
        populations = (
            ("real", step_data.real_patches, step_data.real_scores),
            ("fake", step_data.generated_patches, step_data.fake_scores),
        )
        for population, patches, scores in populations:
            scores = scores.reshape(patches.shape[0], -1).mean(dim=1)
            count = min(self._critic_extreme_patch_count, scores.numel())
            for extreme, largest in (("highest", True), ("lowest", False)):
                selected_scores, indices = scores.topk(count, largest=largest)
                self._writer.add_image(
                    f"images/critic_{population}_{extreme}",
                    make_scored_patch_grid(patches[indices], selected_scores),
                    step_number,
                )


def calculate_sincos_filters(
        normalized_height: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = apply_sin_cos_filters_at_frequencies(normalized_height, SIN_COS_FREQUENCIES)
    height = _display_height(normalized_height)
    high_pass = torch.cat(
        (height, (features[:, 2:3] + 1.0) / 2.0, (features[:, 3:4] + 1.0) / 2.0),
        dim=1,
    )
    low_pass = torch.cat(
        (height, (features[:, 0:1] + 1.0) / 2.0, (features[:, 1:2] + 1.0) / 2.0),
        dim=1,
    )
    return high_pass, low_pass


def calculate_large_map_z0_noise_size(
        z_level: int,
        target_size: int,
        *,
        minimum_noise_size: int = Z0_MINIMUM_NOISE_SIZE,
) -> int:
    if z_level < 0 or target_size <= 16:
        raise ValueError(f"Expected non-negative z_level and target_size > 16, got {z_level}, {target_size}")
    return max(
        minimum_noise_size,
        math.ceil((target_size - 16) / (2 ** z_level) + 32),
    )


def calculate_large_map_input_size(remaining_upscalers: int, target_size: int) -> int:
    if remaining_upscalers < 0 or target_size <= 16:
        raise ValueError(
            f"Expected non-negative remaining_upscalers and target_size > 16, got "
            f"{remaining_upscalers}, {target_size}"
        )
    return math.ceil((target_size - 16) / (2 ** remaining_upscalers) + 16)


def sample_patches_from_regions(
        maps: torch.Tensor,
        *,
        rows: torch.Tensor,
        columns: torch.Tensor,
        region_size: int,
        patch_size: int,
) -> torch.Tensor:
    if rows.shape != (maps.shape[0],) or columns.shape != (maps.shape[0],):
        raise ValueError("rows and columns must provide one output-space coordinate per map")
    valid_offsets = region_size - patch_size + 1
    if valid_offsets <= 0:
        raise ValueError(
            f"Cannot sample {patch_size}x{patch_size} patches from {region_size}x{region_size} regions"
        )
    output_rows = rows + torch.randint(valid_offsets, rows.shape, device=maps.device)
    output_columns = columns + torch.randint(valid_offsets, columns.shape, device=maps.device)
    batch_indices = torch.arange(maps.shape[0], device=maps.device)
    return _extract_patches(maps, batch_indices, output_rows, output_columns, patch_size)


def select_pdf_weighted_patches(
        candidate_patches: torch.Tensor,
        patch_mean_pdf: torch.Tensor,
        *,
        sample_count: int,
        central_fraction: float,
        cap_stddevs: float,
) -> torch.Tensor:
    with torch.no_grad():
        patch_means = candidate_patches.mean(dim=(1, 2, 3))
    return select_pdf_weighted_values(
        patch_means,
        patch_mean_pdf,
        sample_count=sample_count,
        central_fraction=central_fraction,
        cap_stddevs=cap_stddevs,
    )


@lru_cache(maxsize=32)
def _pdf_quantile_probabilities(
        central_fraction: float,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    tail_fraction = (1.0 - central_fraction) / 2.0
    return torch.tensor((tail_fraction, 1.0 - tail_fraction), device=device, dtype=dtype)


def select_pdf_weighted_values(
        candidate_values: torch.Tensor,
        patch_mean_pdf: torch.Tensor,
        *,
        sample_count: int,
        central_fraction: float,
        cap_stddevs: float,
) -> torch.Tensor:
    if sample_count > candidate_values.numel():
        raise ValueError(
            f"sample_count {sample_count} exceeds candidate count {candidate_values.numel()}"
        )
    with torch.no_grad():
        pdf_values = evaluate_patch_mean_pdf(patch_mean_pdf, candidate_values)
        inverse_pdf_weights = pdf_values.clamp_min(torch.finfo(pdf_values.dtype).eps).reciprocal()
        quantile_bounds = torch.quantile(
            inverse_pdf_weights,
            _pdf_quantile_probabilities(
                central_fraction, inverse_pdf_weights.device, inverse_pdf_weights.dtype,
            ),
        )
        central_weights = inverse_pdf_weights[
            (inverse_pdf_weights >= quantile_bounds[0])
            & (inverse_pdf_weights <= quantile_bounds[1])
            ]
        inverse_pdf_weight_cap = central_weights.mean() + cap_stddevs * central_weights.std(
            correction=0
        )
        inverse_pdf_weights = inverse_pdf_weights.clamp_max(inverse_pdf_weight_cap)
        indices = torch.multinomial(inverse_pdf_weights, sample_count, replacement=False)
    return indices


def _extract_patches(
        maps: torch.Tensor,
        batch_indices: torch.Tensor,
        rows: torch.Tensor,
        columns: torch.Tensor,
        patch_size: int,
) -> torch.Tensor:
    patch_offsets = torch.arange(patch_size, device=maps.device)
    patch_rows = rows[:, None, None, None] + patch_offsets[None, None, :, None]
    patch_columns = columns[:, None, None, None] + patch_offsets[None, None, None, :]
    channel_indices = torch.arange(maps.shape[1], device=maps.device)[None, :, None, None]
    return maps[
        batch_indices[:, None, None, None],
        channel_indices,
        patch_rows,
        patch_columns,
    ]


def _make_normalized_patch_logging_image_grid(
        patches: torch.Tensor,
        z_level: int,
        *,
        max_images: int,
        upscale_factor: int,
) -> torch.Tensor:
    patches = patches[:max_images].detach()
    normalized_patches, _ = normalize_downscaled_patches(patches, z_level)
    height = _display_height(patches)
    normalized_values = torch.clamp(
        (normalized_patches + NORMALIZED_PATCH_LIMIT) / (2.0 * NORMALIZED_PATCH_LIMIT),
        0.0,
        1.0,
    )
    rgb_images = torch.cat((height, normalized_values, torch.zeros_like(height)), dim=1)
    return _make_logging_rgb_image_grid(rgb_images, max_images=max_images, upscale_factor=upscale_factor)


def _display_height(normalized_height: torch.Tensor) -> torch.Tensor:
    height = denormalize_elevation(normalized_height)
    return torch.clamp(
        (height - DEFAULT_MIN_ELEV_M) / (DEFAULT_MAX_ELEV_M - DEFAULT_MIN_ELEV_M),
        0.0,
        1.0,
    )


def _assign_gradients(
        parameters: tuple[torch.nn.Parameter, ...],
        gradients: tuple[torch.Tensor, ...],
) -> None:
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient


def _make_logging_rgb_image_grid(
        images: torch.Tensor,
        *,
        max_images: int,
        upscale_factor: int,
) -> torch.Tensor:
    images = F.interpolate(images[:max_images].detach(), scale_factor=upscale_factor, mode="nearest")
    image_count, channels, height, width = images.shape
    column_count = math.ceil(image_count ** 0.5)
    row_count = math.ceil(image_count / column_count)
    padded_image_count = row_count * column_count
    if image_count < padded_image_count:
        empty_images = torch.zeros(
            padded_image_count - image_count,
            channels,
            height,
            width,
            dtype=images.dtype,
            device=images.device,
        )
        images = torch.cat((images, empty_images))

    buffer_size = 4
    images = F.pad(images, (0, buffer_size, 0, buffer_size))
    buffer_height, buffer_width = images.shape[-2:]
    grid = images.view(row_count, column_count, channels, buffer_height, buffer_width)
    grid = grid.permute(2, 0, 3, 1, 4).contiguous()
    return grid.view(channels, row_count * buffer_height, column_count * buffer_width)

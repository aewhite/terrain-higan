from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from terrain_higan.projects.z0_generator.config import default_generator_output_clip_range
from terrain_higan.training import TrainingContext
from terrain_higan.visualization.grids import make_logging_image_grid


@dataclass
class Z0GeneratorStepData:
    """Normalized terrain tensors; real_patches becomes the shifted training batch."""

    real_patches: torch.Tensor | None = None
    generated_norm_field: torch.Tensor | None = None
    generated_norm_patches: torch.Tensor | None = None
    real_patch_shifts: torch.Tensor | None = None
    real_scores: torch.Tensor | None = None
    fake_scores: torch.Tensor | None = None
    generator_adversarial_loss: torch.Tensor | None = None
    generator_elevation_loss: torch.Tensor | None = None
    generator_loss: torch.Tensor | None = None
    discriminator_adversarial_loss: torch.Tensor | None = None
    discriminator_output_mean_loss: torch.Tensor | None = None
    discriminator_loss: torch.Tensor | None = None


def _extract_patches(field: torch.Tensor, patch_size: int, stride: int = 1) -> torch.Tensor:
    unfolded = F.unfold(field, kernel_size=patch_size, stride=stride)
    return unfolded.transpose(1, 2).reshape(-1, 1, patch_size, patch_size)


class GANTrainStage:
    def __init__(
            self,
            *,
            generator: torch.nn.Module,
            discriminator: torch.nn.Module,
            generator_optimizer: torch.optim.Optimizer,
            discriminator_optimizer: torch.optim.Optimizer,
            noise_size: int,
            patch_size: int,
            generator_adversarial_loss_weight: float,
            generator_elevation_loss_weight: float,
            real_patch_shift_stddev: float,
            critic_output_mean_loss_weight: float,
            device: torch.device,
    ) -> None:
        self._generator = generator
        self._discriminator = discriminator
        self._generator_optimizer = generator_optimizer
        self._discriminator_optimizer = discriminator_optimizer
        self._generator_parameters = tuple(generator.parameters())
        self._discriminator_parameters = tuple(discriminator.parameters())
        self._noise_size = noise_size
        self._patch_size = patch_size
        self._generator_adversarial_loss_weight = generator_adversarial_loss_weight
        self._generator_elevation_loss_weight = generator_elevation_loss_weight
        self._real_patch_shift_stddev = real_patch_shift_stddev
        self._critic_output_mean_loss_weight = critic_output_mean_loss_weight
        self._device = device
        self._output_min, self._output_max = default_generator_output_clip_range()

    def run(self, context: TrainingContext[Z0GeneratorStepData]) -> None:
        step_data = context.step_data
        real_patches = step_data.real_patches
        if real_patches is None:
            raise ValueError("GANTrainStage requires real_patches from PatchLoaderStage")

        real_patches, real_patch_shifts = self._shift_real_patches(real_patches)
        random_noise = torch.randn(1, 1, self._noise_size, self._noise_size, device=self._device)
        generated_norm_field = self._generator(random_noise)
        generated_norm_patches = _extract_patches(generated_norm_field, self._patch_size)
        generated_norm_patches = self._sample_patches(
            generated_norm_patches,
            real_patches.shape[0],
        )

        joint_patches = torch.cat((real_patches, generated_norm_patches), dim=0)
        joint_scores = self._discriminator(joint_patches)
        real_scores, fake_scores = torch.split(
            joint_scores,
            (real_patches.shape[0], generated_norm_patches.shape[0]),
        )

        generator_adversarial_loss = F.elu(-fake_scores).mean()
        generator_elevation_loss = self._compute_elevation_loss(
            generated_norm_field,
            real_patches,
        )
        discriminator_adversarial_loss = F.elu(fake_scores).mean() + F.elu(-real_scores).mean()

        discriminator_output_mean_loss = joint_scores.mean().square()
        generator_loss = (
                self._generator_adversarial_loss_weight * generator_adversarial_loss
                + self._generator_elevation_loss_weight * generator_elevation_loss
        )
        discriminator_loss = (
                discriminator_adversarial_loss
                + self._critic_output_mean_loss_weight * discriminator_output_mean_loss
        )

        # Compute both gradients from the same forward pass before updating either model.
        generator_grads = torch.autograd.grad(
            generator_loss,
            self._generator_parameters,
            retain_graph=True,
        )
        discriminator_grads = torch.autograd.grad(
            discriminator_loss,
            self._discriminator_parameters,
        )

        self._assign_grads(self._generator_parameters, generator_grads)
        self._assign_grads(self._discriminator_parameters, discriminator_grads)
        self._generator_optimizer.step()
        self._generator_optimizer.zero_grad(set_to_none=True)
        self._discriminator_optimizer.step()
        self._discriminator_optimizer.zero_grad(set_to_none=True)

        # Publish detached diagnostics from the pre-update forward pass.
        step_data.real_patches = real_patches.detach()
        step_data.generated_norm_field = generated_norm_field.detach()
        step_data.generated_norm_patches = generated_norm_patches.detach()
        step_data.real_patch_shifts = real_patch_shifts.detach()
        step_data.real_scores = real_scores.detach()
        step_data.fake_scores = fake_scores.detach()
        step_data.generator_adversarial_loss = generator_adversarial_loss.detach()
        step_data.generator_elevation_loss = generator_elevation_loss.detach()
        step_data.generator_loss = generator_loss.detach()
        step_data.discriminator_adversarial_loss = discriminator_adversarial_loss.detach()
        step_data.discriminator_output_mean_loss = discriminator_output_mean_loss.detach()
        step_data.discriminator_loss = discriminator_loss.detach()

    def _shift_real_patches(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._real_patch_shift_stddev <= 0.0:
            shifts = torch.zeros(patches.shape[0], 1, 1, 1, device=patches.device, dtype=patches.dtype)
            return patches, shifts

        patch_mins = patches.amin(dim=(1, 2, 3), keepdim=True)
        patch_maxes = patches.amax(dim=(1, 2, 3), keepdim=True)
        min_shifts = self._output_min - patch_mins
        max_shifts = self._output_max - patch_maxes
        shifts = torch.randn_like(min_shifts) * self._real_patch_shift_stddev
        shifts = torch.clamp(shifts, min=min_shifts, max=max_shifts)
        return patches + shifts, shifts

    @staticmethod
    def _sample_patches(patches: torch.Tensor, sample_size: int) -> torch.Tensor:
        if patches.shape[0] < sample_size:
            raise ValueError(f"Need at least {sample_size} patches, got {patches.shape[0]}")
        return GANTrainStage._sample_values(patches, sample_size)

    @staticmethod
    def _compute_elevation_loss(
            generated_norm_field: torch.Tensor,
            real_patches: torch.Tensor,
    ) -> torch.Tensor:
        real_values = real_patches.flatten()
        generated_values = generated_norm_field.flatten()
        sample_size = min(real_values.shape[0], generated_values.shape[0])
        if sample_size <= 0:
            raise ValueError("Need at least one real and generated elevation value for elevation loss")

        real_values = GANTrainStage._sample_values(real_values, sample_size)
        generated_values = GANTrainStage._sample_values(generated_values, sample_size)
        squared_distances = (generated_values.sort().values - real_values.sort().values).square()
        return squared_distances.mean() + squared_distances.std(unbiased=False)

    @staticmethod
    def _sample_values(values: torch.Tensor, sample_size: int) -> torch.Tensor:
        if values.shape[0] == sample_size:
            return values
        indices = torch.randperm(values.shape[0], device=values.device)[:sample_size]
        return values[indices]

    @staticmethod
    def _assign_grads(
            parameters: tuple[torch.nn.Parameter, ...],
            grads: tuple[torch.Tensor, ...],
    ) -> None:
        for parameter, grad in zip(parameters, grads):
            parameter.grad = grad


class TensorBoardLogStage:
    def __init__(
            self,
            *,
            writer: SummaryWriter,
            patch_size: int,
            log_every_steps: int,
            image_every_steps: int,
    ) -> None:
        self._writer = writer
        self._patch_size = patch_size
        self._log_every_steps = log_every_steps
        self._image_every_steps = image_every_steps
        self._generator_losses: list[torch.Tensor] = []
        self._critic_losses: list[torch.Tensor] = []

    def run(self, context: TrainingContext[Z0GeneratorStepData]) -> None:
        step_number = context.step_number
        step_data = context.step_data
        if step_data.generator_loss is not None:
            self._generator_losses.append(step_data.generator_loss.detach())
        if step_data.discriminator_loss is not None:
            self._critic_losses.append(step_data.discriminator_loss.detach())

        if step_number % self._log_every_steps == 0:
            self._log_scalars(context)
            self._log_histograms(context)
            self._writer.flush()

        if step_number % self._image_every_steps == 0:
            self._log_images(context)

    def _log_scalars(self, context: TrainingContext[Z0GeneratorStepData]) -> None:
        step_number = context.step_number
        step_data = context.step_data

        for tag, loss in (
                ("loss/generator", step_data.generator_loss),
                ("loss/generator_adversarial", step_data.generator_adversarial_loss),
                ("loss/generator_elevation", step_data.generator_elevation_loss),
                ("loss/critic", step_data.discriminator_loss),
                ("loss/critic_adversarial", step_data.discriminator_adversarial_loss),
                ("loss/critic_output_mean", step_data.discriminator_output_mean_loss),
        ):
            if loss is not None:
                self._writer.add_scalar(tag, loss.item(), step_number)
        if step_data.real_scores is not None:
            self._writer.add_scalar("scores/critic_real_mean", step_data.real_scores.mean().item(), step_number)
        if step_data.fake_scores is not None:
            self._writer.add_scalar("scores/critic_fake_mean", step_data.fake_scores.mean().item(), step_number)

    def _log_histograms(self, context: TrainingContext[Z0GeneratorStepData]) -> None:
        step_number = context.step_number
        step_data = context.step_data

        for tag, values in (
                ("elevations/real", step_data.real_patches),
                ("elevations/generated", step_data.generated_norm_field),
                ("scores/critic_real", step_data.real_scores),
                ("scores/critic_fake", step_data.fake_scores),
        ):
            if values is not None:
                self._writer.add_histogram(tag, values, step_number)
        if self._generator_losses:
            self._writer.add_histogram("loss/generator_histogram", torch.stack(self._generator_losses), step_number)
            self._generator_losses.clear()
        if self._critic_losses:
            self._writer.add_histogram("loss/critic_histogram", torch.stack(self._critic_losses), step_number)
            self._critic_losses.clear()

    def _log_images(self, context: TrainingContext[Z0GeneratorStepData]) -> None:
        step_number = context.step_number
        step_data = context.step_data

        generated_logging_patches = None
        if step_data.generated_norm_field is not None:
            generated_logging_patches = _extract_patches(
                step_data.generated_norm_field.detach(),
                self._patch_size,
                stride=self._patch_size,
            )

        for tag, images, max_images, upscale_factor in (
                ("images/real_patches", step_data.real_patches, 64, 24),
                ("images/generated_patches", generated_logging_patches, 64, 24),
                ("images/output_maps", step_data.generated_norm_field, 1, 4),
        ):
            if images is not None:
                self._writer.add_image(
                    tag,
                    make_logging_image_grid(
                        images,
                        max_images=max_images,
                        upscale_factor=upscale_factor,
                    ).cpu(),
                    step_number,
                )

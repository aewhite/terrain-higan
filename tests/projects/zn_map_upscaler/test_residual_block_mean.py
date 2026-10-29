import unittest
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from terrain_higan.etopo.normalization import (
    ELEVATION_STD_M,
    SYNTHETIC_STD_SCALES,
    UPSCALE_STATS,
    denormalize_residuals,
    normalize_residuals,
)
from terrain_higan.models.upscaling import mean_preserving_bilinear_2x
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.stages import (
    ScaleRealPatchesStage,
    ZnUpscalerStepData,
    ZnUpscalerTensorBoardStage,
    residual_block_mean_loss,
)
from terrain_higan.training import TrainingContext


class ResidualBlockMeanTest(unittest.TestCase):
    def test_zero_sum_detail_and_opposing_local_biases(self) -> None:
        detail = torch.tensor([[[[3.0, -1.0], [-1.0, -1.0]]]])
        self.assertEqual(residual_block_mean_loss(detail, 0.5).item(), 0.0)
        biased = torch.cat((detail + 1.0, detail - 1.0), dim=-1)
        self.assertEqual(biased.mean().item(), 0.0)
        self.assertEqual(residual_block_mean_loss(biased, 0.5).item(), 0.0)
        repeated = biased.repeat(3, 1, 2, 4)
        torch.testing.assert_close(residual_block_mean_loss(repeated, 0.5), torch.tensor(0.0))

    def test_opposing_patch_biases_do_not_cancel_across_batch(self) -> None:
        residuals = torch.stack((torch.ones(1, 4, 6), -torch.ones(1, 4, 6)))
        residuals.requires_grad_()
        loss = residual_block_mean_loss(residuals, 0.5)
        self.assertEqual(loss.item(), 4.0)
        gradient, = torch.autograd.grad(loss, residuals)
        torch.testing.assert_close(gradient[0], torch.full((1, 4, 6), 1 / 6))
        torch.testing.assert_close(gradient[1], torch.full((1, 4, 6), -1 / 6))
        repeated = residuals.detach().repeat(3, 1, 2, 4)
        self.assertEqual(residual_block_mean_loss(repeated, 0.5).item(), 4.0)

    def test_loss_uses_linear_residuals_after_inverse_sinh(self) -> None:
        transformed = torch.tensor([[[[2.0, -1.0], [-1.0, 0.0]]]])
        self.assertEqual(transformed.sum().item(), 0.0)
        divisor = UPSCALE_STATS[5].residual_divisor
        residuals = denormalize_residuals(transformed, 5)
        self.assertGreater(residual_block_mean_loss(residuals, divisor).item(), 0.0)
        linear = torch.tensor([[[[3.0, -1.0], [-1.0, -1.0]]]]) * divisor
        self.assertNotEqual(normalize_residuals(linear, 5).sum().item(), 0.0)
        torch.testing.assert_close(residual_block_mean_loss(linear, divisor), torch.tensor(0.0), atol=1e-12, rtol=0)

    def test_zoom_scaling_and_gradients_through_sinh(self) -> None:
        reference = torch.tensor([[[[2.0, -1.0], [-1.0, 0.0]]]])
        losses, gradients = [], []
        for z_level in (5, 10, 20):
            transformed = reference.clone().requires_grad_()
            residuals = denormalize_residuals(transformed, z_level)
            loss = residual_block_mean_loss(residuals, UPSCALE_STATS[z_level].residual_divisor)
            gradient, = torch.autograd.grad(loss, transformed)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue((gradient != 0).all())
            expected_gradient = reference.sinh().mean() * reference.cosh() / 2
            torch.testing.assert_close(gradient, expected_gradient)
            losses.append(loss)
            gradients.append(gradient)
        for loss, gradient in zip(losses[1:], gradients[1:]):
            torch.testing.assert_close(loss, losses[0])
            torch.testing.assert_close(gradient, gradients[0])

    def test_rectangular_output_alignment_and_meter_diagnostics_after_clipping(self) -> None:
        model = ZnUpscalerModel(z_level=20, layer_channels=(8, 8, 4, 2))
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model._network[-1].bias.fill_(1.0)
        input_map = torch.linspace(0.0, 0.02, 12 * 15).reshape(1, 1, 12, 15)
        input_map[..., 4, 4] = 2 * UPSCALE_STATS[20].dataset_maximum_value
        residuals, output_map = model(input_map, torch.zeros(1, 4, 12, 15)).split(1, dim=1)
        parent_map = input_map[..., 4:-4, 4:-4]
        learned_residual = denormalize_residuals(torch.tensor(1.0), 20)
        expected_parent = mean_preserving_bilinear_2x(input_map)[..., 8:-8, 8:-8]
        reference_parent = F.interpolate(input_map, scale_factor=2, mode="nearest")[..., 8:-8, 8:-8]
        expected_residuals = learned_residual + (expected_parent - reference_parent)
        torch.testing.assert_close(residuals, expected_residuals, rtol=0, atol=0)
        corrected_residuals = expected_residuals.clamp(
            UPSCALE_STATS[20].dataset_minimum_raw_residual,
            UPSCALE_STATS[20].dataset_maximum_raw_residual,
        )
        expected_output = (reference_parent + corrected_residuals).clamp(
            UPSCALE_STATS[20].dataset_minimum_value,
            UPSCALE_STATS[20].dataset_maximum_value,
        )
        expected_drift = F.avg_pool2d(expected_output, 2, 2) - parent_map
        torch.testing.assert_close(F.avg_pool2d(output_map, 2, 2) - parent_map, expected_drift, atol=1e-8, rtol=1e-4)

        data = ZnUpscalerStepData()
        data.input_map = input_map
        data.input_metric_patches = parent_map
        data.denormalized_residuals = residuals.detach()
        data.output_map = output_map.detach()
        data.real_residual_block_means = torch.tensor([[[[2.0, -2.0]]]]) * UPSCALE_STATS[20].residual_divisor
        for name in (
                "generator_loss", "generator_adversarial_loss", "generator_residual_aggregate_loss",
                "generator_residual_bounds_loss", "generator_residual_block_mean_loss",
                "generator_residual_block_mean_loss_weighted", "critic_loss",
        ):
            setattr(data, name, torch.tensor(0.0))
        writer = Mock()
        logger = ZnUpscalerTensorBoardStage(writer, z_level=20, log_every_steps=1, image_every_steps=100)
        logger._log_scalars(data, 1)
        values = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
        expected_block_means_meters = F.avg_pool2d(expected_residuals, 2, 2) * ELEVATION_STD_M
        self.assertEqual(values["residuals/block_mean_meters_mean"], expected_block_means_meters.mean().item())
        self.assertEqual(values["residuals/block_mean_meters_rms"], expected_block_means_meters.square().mean().sqrt().item())
        drift_meters = (F.avg_pool2d(data.output_map, 2, 2) - parent_map) * ELEVATION_STD_M
        self.assertEqual(values["consistency/output_parent_drift_meters_mean"], drift_meters.mean().item())
        self.assertEqual(values["consistency/output_parent_drift_meters_rms"], drift_meters.square().mean().sqrt().item())
        self.assertEqual(values["residuals/real_block_mean_scaled_rms"], 2.0)
        self.assertLess(values["consistency/output_parent_drift_meters_mean"], values["residuals/block_mean_meters_mean"])

    def test_synthetic_targets_change_parent_means(self) -> None:
        patches = torch.arange(100, dtype=torch.float64).reshape(1, 1, 10, 10) / 100
        for z_level in (11, 12, 20):
            data = ZnUpscalerStepData()
            data.real_patches = patches.clone()
            ScaleRealPatchesStage(z_level).run(TrainingContext(step_number=1, step_data=data))
            source_means = F.avg_pool2d(data.real_source_patches, 2, 2)
            upscaled_source = F.interpolate(source_means, scale_factor=2, mode="nearest")
            residuals = data.real_patches - upscaled_source
            previous_scale = 1.0 if z_level == 11 else SYNTHETIC_STD_SCALES[z_level - 12]
            current_scale = SYNTHETIC_STD_SCALES[z_level - 11]
            expected = (current_scale - previous_scale) * (F.avg_pool2d(patches, 2, 2) - patches.mean())
            torch.testing.assert_close(F.avg_pool2d(residuals, 2, 2), expected, atol=1e-7, rtol=1e-4)
            torch.testing.assert_close(
                residual_block_mean_loss(residuals, UPSCALE_STATS[z_level].residual_divisor),
                torch.tensor(0.0, dtype=patches.dtype), atol=1e-10, rtol=0,
            )

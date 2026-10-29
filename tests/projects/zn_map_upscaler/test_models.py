import unittest
import pytest

import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from terrain_higan.etopo.normalization import UPSCALE_STATS
from terrain_higan.inference.models import Upscaler
from terrain_higan.models.upscaling import mean_preserving_bilinear_2x
from terrain_higan.projects.zn_map_upscaler.models import SIN_COS_FREQUENCIES, ZnUpscalerCritic, ZnUpscalerModel


class ZnUpscalerModelTest(unittest.TestCase):
    def test_sinh_input_clipping_keeps_outputs_and_corrective_gradients_finite(self) -> None:
        input_map = torch.zeros(1, 1, 12, 15)
        noise = torch.zeros(1, 4, 12, 15)
        for z_level in (1, 20):
            for value in (-100.0, 100.0):
                with self.subTest(z_level=z_level, value=value):
                    model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2))
                    with torch.no_grad():
                        for parameter in model.parameters():
                            parameter.zero_()
                        model._network[-1].bias.fill_(value)
                    residuals, output = model(input_map, noise).split(1, dim=1)
                    expected = torch.tensor(value).clamp(-5, 5).sinh() * UPSCALE_STATS[z_level].residual_divisor
                    torch.testing.assert_close(residuals, torch.full_like(residuals, expected))
                    self.assertTrue(torch.isfinite(output).all())
                    residuals.square().mean().backward()
                    self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
                    self.assertGreater(value * model._network[-1].bias.grad.item(), 0)
                    inference = Upscaler(checkpoint_zoom=z_level, layer_channels=(8, 8, 4, 2))
                    inference.load_state_dict(model.state_dict())
                    torch.testing.assert_close(
                        inference(input_map, noise, destination_zoom=z_level), output,
                        rtol=0, atol=1e-6,
                    )

    def test_network_features_remain_coarse_until_the_middle_shuffle(self) -> None:
        input_map = torch.zeros(1, 1, 12, 15)
        input_map[..., 5, 6] = 0.01
        noise = torch.randn(1, 4, 12, 15) * 4
        nearest = F.interpolate(input_map, scale_factor=2, mode="nearest")
        base = mean_preserving_bilinear_2x(input_map)
        self.assertGreater((base - nearest).abs().max().item(), 0)
        expected_features = [input_map]
        for frequency in SIN_COS_FREQUENCIES:
            phase = frequency * 2.0 * torch.pi * input_map
            expected_features.extend((phase.sin(), phase.cos()))
        expected_features = torch.cat(expected_features, dim=1)
        expected_features = torch.cat((expected_features, noise.clamp(-5, 5)), dim=1)
        for z_level in (1, 10, 11, 20):
            with self.subTest(z_level=z_level):
                training = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2))
                inference = Upscaler(checkpoint_zoom=z_level, layer_channels=(8, 8, 4, 2))
                inference.load_state_dict(training.state_dict())
                observed = []
                for model in (training, inference):
                    model._network.register_forward_pre_hook(
                        lambda module, inputs: observed.append(inputs[0].detach()),
                    )
                _, training_output = training(input_map, noise).split(1, dim=1)
                inference_output = inference(input_map, noise, destination_zoom=z_level)
                for features in observed:
                    torch.testing.assert_close(features, expected_features, rtol=0, atol=0)
                torch.testing.assert_close(inference_output, training_output, rtol=0, atol=1e-6)

    def test_training_and_evaluation_outputs(self) -> None:
        for z_level in (1, 10, *range(11, 21)):
            with self.subTest(z_level=z_level):
                model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2))
                input_map = torch.randn(2, 1, 12, 15) * 0.01
                noise = torch.randn(2, 4, 12, 15)
                training_maps = model(input_map, noise)
                self.assertEqual(training_maps.shape, (2, 2, 8, 14))
                residuals, output_map = training_maps.split(1, dim=1)
                self.assertEqual(residuals.shape, (2, 1, 8, 14))
                stats = UPSCALE_STATS[z_level]
                upscaled_input = F.interpolate(input_map, scale_factor=2, mode="nearest")
                expected = (
                        upscaled_input[..., 8:-8, 8:-8]
                        + residuals.clamp(stats.dataset_minimum_raw_residual, stats.dataset_maximum_raw_residual)
                ).clamp(stats.dataset_minimum_value, stats.dataset_maximum_value)
                torch.testing.assert_close(output_map, expected, rtol=0, atol=0)
                model.eval()
                torch.testing.assert_close(model(input_map, noise), output_map, rtol=0, atol=0)
                self.assertEqual(model._network[0][0].in_channels, 9)
                self.assertIn("_network.0.0.weight", model.state_dict())
                self.assertIn("_network.1.weight", model.state_dict())

    def test_residual_includes_mean_preserving_detail_and_balance_gradient(self) -> None:
        input_map = torch.zeros(1, 1, 12, 15)
        input_map[..., 5, 6] = 0.001
        noise = torch.zeros(1, 4, 12, 15)
        nearest_parent = F.interpolate(input_map, scale_factor=2, mode="nearest")[..., 8:-8, 8:-8]
        bilinear_parent = mean_preserving_bilinear_2x(input_map)[..., 8:-8, 8:-8]
        for z_level in range(1, 21):
            with self.subTest(z_level=z_level):
                model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2))
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                network_outputs = []

                def capture_network_output(module, inputs, output):
                    output.retain_grad()
                    network_outputs.append(output)

                model._network.register_forward_hook(capture_network_output)
                residuals, output_map = model(input_map, noise).split(1, dim=1)
                expected_parent = bilinear_parent
                expected_residual = expected_parent - nearest_parent
                torch.testing.assert_close(residuals, expected_residual, rtol=0, atol=0)
                torch.testing.assert_close(output_map, expected_parent, rtol=0, atol=1e-10)
                block_means = F.avg_pool2d(residuals, 2, 2)
                torch.testing.assert_close(block_means, torch.zeros_like(block_means), rtol=0, atol=1e-10)
                with torch.no_grad():
                    model._network[-1].bias.fill_(0.25)
                residuals, _ = model(input_map, noise).split(1, dim=1)
                loss = (F.avg_pool2d(residuals, 2, 2) / UPSCALE_STATS[z_level].residual_divisor).square().mean()
                loss.backward()
                self.assertGreater(loss.item(), 0.0)
                self.assertTrue(bool((network_outputs[-1].grad != 0).any()))

    @pytest.mark.export
    def test_dynamic_model_export(self) -> None:
        model = ZnUpscalerModel(z_level=20, layer_channels=(8, 8, 4, 2)).eval()
        with torch.no_grad():
            model._network[-1].bias.fill_(100.0)
        batch = torch.export.Dim("batch", min=1)
        height = torch.export.Dim("height", min=9)
        width = torch.export.Dim("width", min=9)
        spatial_dims = {0: batch, 2: height, 3: width}
        exported = torch.export.export(
            model,
            (torch.randn(2, 1, 32, 32), torch.randn(2, 4, 32, 32)),
            dynamic_shapes={"input_map": spatial_dims, "noise": spatial_dims},
        ).module()
        with torch.inference_mode():
            for batch_size, h, w in ((1, 13, 15), (3, 20, 18)):
                x = torch.randn(batch_size, 1, h, w)
                noise = torch.randn(batch_size, 4, h, w)
                torch.testing.assert_close(exported(x, noise), model(x, noise))

    def test_critic_retains_spatial_scores(self) -> None:
        for layer_specs, patch_size, score_count in (("4/3 2/1", 4, 4), ("4/1", 4, 16)):
            with self.subTest(layer_specs=layer_specs):
                critic = ZnUpscalerCritic(z_level=1, layer_specs=layer_specs)
                scores = critic(torch.randn(3, 1, patch_size, patch_size))
                self.assertEqual(scores.shape, (3, score_count))
                self.assertTrue(torch.all((scores >= -10) & (scores <= 10)))
                self.assertTrue(any(key.startswith("_first_layer.") for key in critic.state_dict()))

    def test_critic_normalizes_feature_weights_but_preserves_head_and_context(self) -> None:
        for layer_specs in ("4/3 2/1", "4/1"):
            with self.subTest(layer_specs=layer_specs):
                critic = ZnUpscalerCritic(z_level=1, layer_specs=layer_specs).eval()
                head = critic._network[-1]
                context = critic._first_layer.linear
                self.assertFalse(parametrize.is_parametrized(head, "weight"))
                self.assertFalse(parametrize.is_parametrized(context, "weight"))
                self.assertEqual(context.weight.count_nonzero().item(), 0)
                self.assertEqual(context.bias.count_nonzero().item(), 0)
                for layer in critic.modules():
                    if not isinstance(layer, torch.nn.Conv2d) or layer is head:
                        continue
                    self.assertTrue(parametrize.is_parametrized(layer, "weight"))
                    weight = layer.weight.detach().clone()
                    with torch.no_grad():
                        layer.parametrizations.weight.original.mul_(3.0)
                    torch.testing.assert_close(layer.weight, weight)
                patches = torch.randn(3, 1, 4, 4, requires_grad=True)
                scores = critic(patches)
                scores.square().mean().backward()
                self.assertTrue(torch.isfinite(scores).all())
                self.assertTrue(torch.isfinite(patches.grad).all())
                self.assertGreater(patches.grad.abs().sum().item(), 0)
                for parameter in critic.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_critic_spectral_normalization_state_round_trip(self) -> None:
        critic = ZnUpscalerCritic(z_level=1, layer_specs="4/3 2/1")
        patches = torch.randn(3, 1, 4, 4)
        critic(patches)
        critic.eval()
        restored = ZnUpscalerCritic(z_level=1, layer_specs="4/3 2/1").eval()
        restored.load_state_dict(critic.state_dict())
        torch.testing.assert_close(restored(patches), critic(patches), rtol=0, atol=0)

    def test_noise_clipping(self) -> None:
        model = ZnUpscalerModel(z_level=1, layer_channels=(8, 8, 4, 2)).eval()
        input_map = torch.zeros(1, 1, 12, 12)
        for value in (-6.0, 6.0):
            noise = torch.full((1, 4, 12, 12), value)
            torch.testing.assert_close(model(input_map, noise), model(input_map, noise.clamp(-5, 5)))

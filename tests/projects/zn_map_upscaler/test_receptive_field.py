import unittest

import torch

from terrain_higan.inference.models import Upscaler
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel


def _active_models():
    for name, model, arguments in (
        ("training", ZnUpscalerModel(z_level=1, layer_channels=(4, 4, 2, 2)), {}),
        ("inference", Upscaler(checkpoint_zoom=1, layer_channels=(4, 4, 2, 2)).eval(), {"destination_zoom": 1}),
    ):
        model = model.double()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(0.01)
        yield name, model, arguments


class UpscalerReceptiveFieldTest(unittest.TestCase):
    def test_valid_stride_one_layers_and_midpoint_shapes(self) -> None:
        network = ZnUpscalerModel(z_level=1, layer_channels=(4, 4, 2, 2))._network
        for layer in network.modules():
            if isinstance(layer, torch.nn.Conv2d):
                self.assertEqual(layer.stride, (1, 1))
                self.assertEqual(layer.padding, (0, 0))
                self.assertEqual(layer.kernel_size, (1, 1) if layer is network[-1] else (3, 3))
        coarse = network[0][:4](torch.ones(1, 9, 15, 17))
        self.assertEqual(coarse.shape, (1, 4, 9, 11))
        shuffled = network[0][4](coarse)
        self.assertEqual(shuffled.shape, (1, 1, 18, 22))
        output = network(torch.ones(1, 9, 15, 17))
        self.assertEqual(output.shape, (1, 1, 14, 18))

    def test_shuffle_channel_width_must_be_divisible_by_four(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            ZnUpscalerModel(z_level=1, layer_channels=(4, 6, 2, 2))

    def test_exact_parent_and_noise_support_for_every_output_phase(self) -> None:
        for name, model, arguments in _active_models():
            parents = torch.zeros(1, 1, 15, 17, dtype=torch.double, requires_grad=True)
            noise = torch.zeros(1, 4, 15, 17, dtype=torch.double, requires_grad=True)
            output = model(parents, noise, **arguments)
            for channel in range(output.shape[1]):
                for row in (2, 3):
                    for column in (4, 5):
                        with self.subTest(model=name, channel=channel, phase=(row % 2, column % 2)):
                            parent_gradient, noise_gradient = torch.autograd.grad(
                                output[0, channel, row, column], (parents, noise), retain_graph=True,
                            )
                            parent_support = torch.zeros_like(parents, dtype=torch.bool)
                            parent_support[..., row // 2:row // 2 + 9, column // 2:column // 2 + 9] = True
                            self.assertTrue(torch.equal(parent_gradient != 0, parent_support))
                            self.assertTrue(torch.equal(noise_gradient != 0, parent_support.expand_as(noise_gradient)))

    def test_outside_perturbations_cannot_change_any_output_phase(self) -> None:
        for name, model, arguments in _active_models():
            parents = torch.zeros(1, 1, 15, 17, dtype=torch.double)
            noise = torch.zeros(1, 4, 15, 17, dtype=torch.double)
            with torch.inference_mode():
                expected = model(parents, noise, **arguments)
                for row in (2, 3):
                    for column in (4, 5):
                        parent_support = torch.zeros_like(parents, dtype=torch.bool)
                        parent_support[..., row // 2:row // 2 + 9, column // 2:column // 2 + 9] = True
                        changed_parents = torch.where(parent_support, parents, 0.02)
                        changed_noise = torch.where(parent_support.expand_as(noise), noise, -0.02)
                        for change, parent_input, noise_input in (
                            ("parent", changed_parents, noise),
                            ("noise", parents, changed_noise),
                            ("both", changed_parents, changed_noise),
                        ):
                            with self.subTest(model=name, change=change, phase=(row % 2, column % 2)):
                                actual = model(parent_input, noise_input, **arguments)
                                torch.testing.assert_close(
                                    actual[..., row, column], expected[..., row, column], rtol=0, atol=0,
                                )

    def test_overlapping_parent_crops_keep_global_registration(self) -> None:
        torch.manual_seed(17)
        for name, model, arguments in _active_models():
            parents = torch.randn(1, 1, 31, 35, dtype=torch.double) * 0.01
            noise = torch.randn(1, 4, 31, 35, dtype=torch.double) * 0.01
            with torch.inference_mode():
                full = model(parents, noise, **arguments)
                for row, column in ((3, 5), (4, 6)):
                    with self.subTest(model=name, origin=(row, column)):
                        patch = model(
                            parents[..., row:row + 21, column:column + 23],
                            noise[..., row:row + 21, column:column + 23], **arguments,
                        )
                        expected = full[..., 2 * row:2 * row + 26, 2 * column:2 * column + 30]
                        torch.testing.assert_close(patch, expected, rtol=1e-10, atol=1e-10)

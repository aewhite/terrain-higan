import unittest

import torch
from torch.nn.utils import parametrize

from terrain_higan.models import deep_conv2d


class DeepConv2dTest(unittest.TestCase):
    def test_per_layer_dilations_preserve_dense_receptive_fields(self) -> None:
        network = deep_conv2d(1, "2/3 2/3/2 2/3/2 2/3/2 1/3")
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.fill_(1.0)
        patches = torch.ones(1, 1, 17, 17, requires_grad=True)
        features = patches
        receptive_field = 1
        for layer in network:
            features = layer(features)
            if not isinstance(layer, torch.nn.Conv2d):
                continue
            receptive_field += (layer.kernel_size[0] - 1) * layer.dilation[0]
            self.assertEqual(features.shape[-2:], (18 - receptive_field,) * 2)
            gradient = torch.autograd.grad(features[0, 0, 0, 0], patches, retain_graph=True)[0]
            expected_support = torch.zeros_like(patches, dtype=torch.bool)
            expected_support[..., :receptive_field, :receptive_field] = True
            self.assertTrue(torch.equal(gradient != 0, expected_support))
        self.assertEqual(receptive_field, 17)

    def test_invalid_convolution_or_shuffle_specs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid Conv2d layer spec"):
            deep_conv2d(1, "2/3/1/1")
        with self.assertRaisesRegex(ValueError, "Invalid shuffle spec"):
            deep_conv2d(4, "shuffle/2/2")
        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            deep_conv2d(6, "shuffle/2")

    def test_shuffle_spec_tracks_channels_shapes_and_gradients(self) -> None:
        network = deep_conv2d(9, "4/3 4/3/2 shuffle/2 2/3 2/3")
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.fill_(0.01)
        inputs = torch.ones(2, 9, 15, 17, requires_grad=True)
        output = network(inputs)
        self.assertIsInstance(network[4], torch.nn.PixelShuffle)
        self.assertEqual(network[5].in_channels, 1)
        self.assertEqual(output.shape, (2, 2, 14, 18))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(inputs.grad.abs().sum().item(), 0)
        for parameter in network.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_spectral_normalization_preserves_conv_options_and_gradients(self) -> None:
        for spectral_norm in (False, True):
            with self.subTest(spectral_norm=spectral_norm):
                network = deep_conv2d(
                    2, "4/3 3/1", activation=torch.nn.SiLU,
                    spectral_norm=spectral_norm, bias=False, stride=2,
                )
                patches = torch.randn(2, 2, 9, 9, requires_grad=True)
                output = network(patches)
                self.assertEqual(output.shape, (2, 3, 2, 2))
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(patches.grad).all())
                self.assertGreater(patches.grad.abs().sum().item(), 0)
                for layer in network:
                    if isinstance(layer, torch.nn.Conv2d):
                        self.assertIsNone(layer.bias)
                        self.assertEqual(parametrize.is_parametrized(layer, "weight"), spectral_norm)
                for parameter in network.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_default_keeps_plain_weights_and_normalization_removes_weight_scale(self) -> None:
        plain = deep_conv2d(1, "4/3 2/1")
        explicit_plain = deep_conv2d(1, "4/3 2/1", spectral_norm=False)
        explicit_plain.load_state_dict(plain.state_dict())
        patches = torch.randn(2, 1, 5, 5)
        torch.testing.assert_close(plain(patches), explicit_plain(patches), rtol=0, atol=0)
        for layer in plain:
            self.assertFalse(parametrize.is_parametrized(layer, "weight"))

        normalized = deep_conv2d(1, "4/3 2/1", spectral_norm=True).eval()
        expected = normalized(patches).detach()
        with torch.no_grad():
            for layer in normalized:
                if isinstance(layer, torch.nn.Conv2d):
                    layer.parametrizations.weight.original.mul_(3.0)
        torch.testing.assert_close(normalized(patches), expected)

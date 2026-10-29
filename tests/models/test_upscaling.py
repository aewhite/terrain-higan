import unittest
import pytest

import torch
import torch.nn.functional as F

from terrain_higan.models.upscaling import mean_preserving_bilinear_2x
from terrain_higan.visualization.grids import _make_grid


class MeanPreservingUpscalingTest(unittest.TestCase):
    def test_local_means_shapes_devices_and_dtypes(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device in devices:
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for shape in ((2, 1, 5, 7), (2, 1, 2, 2), (1, 1, 3, 6), (2, 1, 7, 4)):
                    with self.subTest(device=device, dtype=dtype, shape=shape):
                        x = torch.randn(shape, device=device, dtype=dtype)
                        y = mean_preserving_bilinear_2x(x)
                        self.assertEqual(y.shape, (*shape[:2], 2 * shape[2], 2 * shape[3]))
                        self.assertEqual(y.dtype, x.dtype)
                        self.assertEqual(y.device, x.device)
                        eps = torch.finfo(dtype).eps
                        torch.testing.assert_close(F.avg_pool2d(y, 2, 2), x, rtol=8 * eps, atol=8 * eps)

    def test_constant_ramp_and_boundary_impulses(self):
        ramp = torch.arange(35, dtype=torch.float64).reshape(1, 1, 5, 7)
        impulses = torch.zeros(4, 1, 5, 7, dtype=torch.float64)
        impulses[0, 0, 0, 0] = 1
        impulses[1, 0, 0, -1] = 1
        impulses[2, 0, -1, 0] = 1
        impulses[3, 0, -1, -1] = 1
        for x in (ramp, impulses, torch.full_like(ramp, 2.5)):
            y = mean_preserving_bilinear_2x(x)
            torch.testing.assert_close(F.avg_pool2d(y, 2, 2), x, rtol=0, atol=1e-14)
        torch.testing.assert_close(mean_preserving_bilinear_2x(torch.full_like(ramp, 2.5)),
                                   torch.full((1, 1, 10, 14), 2.5, dtype=torch.float64))

    def test_central_and_one_sided_differences(self):
        x = (torch.tensor([0.0, 8.0, -2.0])[:, None]
             + torch.tensor([1.0, 4.0, 2.0, 8.0])[None, :]).reshape(1, 1, 3, 4)
        gx = torch.tensor([3.0, 0.5, 2.0, 6.0]).reshape(1, 1, 1, 4)
        gy = torch.tensor([8.0, -1.0, -10.0]).reshape(1, 1, 3, 1)
        y = mean_preserving_bilinear_2x(x)
        for row, column in ((0, 0), (0, 1), (1, 0), (1, 1)):
            expected = x + (2 * column - 1) * gx / 4 + (2 * row - 1) * gy / 4
            torch.testing.assert_close(y[..., row::2, column::2], expected, rtol=0, atol=0)

    def test_linear_slopes_including_diagonals_and_boundaries(self):
        for height, width in ((5, 7), (2, 2), (3, 6), (7, 4)):
            for slope_y, slope_x in ((0.0, 1.25), (3.5, 0.0), (3.5, -1.25)):
                with self.subTest(shape=(height, width), slopes=(slope_y, slope_x)):
                    offsets = torch.arange(3, dtype=torch.float64).reshape(3, 1, 1, 1)
                    rows = torch.arange(height, dtype=torch.float64).reshape(1, 1, height, 1)
                    columns = torch.arange(width, dtype=torch.float64).reshape(1, 1, 1, width)
                    x = offsets + slope_y * rows + slope_x * columns
                    child_rows = (torch.arange(2 * height, dtype=torch.float64) / 2 - 0.25).reshape(1, 1, 2 * height, 1)
                    child_columns = (torch.arange(2 * width, dtype=torch.float64) / 2 - 0.25).reshape(1, 1, 1, 2 * width)
                    expected = offsets + slope_y * child_rows + slope_x * child_columns
                    torch.testing.assert_close(mean_preserving_bilinear_2x(x), expected, rtol=0, atol=0)

    def test_noncontiguous_inputs(self):
        x = torch.randn(2, 1, 5, 7, dtype=torch.float64)
        for values in (x.transpose(-1, -2), x.to(memory_format=torch.channels_last)):
            torch.testing.assert_close(F.avg_pool2d(mean_preserving_bilinear_2x(values), 2, 2), values)

    def test_gradients_and_conserved_mean_gradient(self):
        x = torch.randn(1, 1, 2, 3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(mean_preserving_bilinear_2x, (x,)))
        weights = torch.randn_like(x)
        pooled = F.avg_pool2d(mean_preserving_bilinear_2x(x), 2, 2)
        gradient, = torch.autograd.grad((pooled * weights).sum(), x)
        torch.testing.assert_close(gradient, weights, rtol=0, atol=1e-14)

    @pytest.mark.export
    def test_compile_and_export(self):
        class Upscale(torch.nn.Module):
            def forward(self, x):
                return mean_preserving_bilinear_2x(x)

        x = torch.randn(2, 1, 5, 7)
        compiled = torch.compile(mean_preserving_bilinear_2x, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), mean_preserving_bilinear_2x(x))
        exported = torch.export.export(Upscale(), (x,)).module()
        torch.testing.assert_close(exported(x), mean_preserving_bilinear_2x(x))

    def test_preview_preserves_discrete_pixels_at_every_scale(self):
        x = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        for scale in (1, 2, 3, 4, 6, 16, 24):
            with self.subTest(scale=scale):
                actual = _make_grid(x, 1, 1, scale, 0, colorize=lambda values: values)
                expected = x.repeat_interleave(scale, -2).repeat_interleave(scale, -1)
                torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)

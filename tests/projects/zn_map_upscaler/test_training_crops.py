import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as F

from terrain_higan.etopo.normalization import ELEVATION_STD_M, UPSCALE_STATS, normalize_residuals
from terrain_higan.projects.z0_generator.export import EXPORT_MINIMUM_INPUT_SIZE, Z0GeneratorExportModule
from terrain_higan.projects.z0_generator.models import Z0Generator
from terrain_higan.projects.zn_map_upscaler.geometry import center_crop
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerCritic, ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.stages import (
    MaterializedInputMapStage,
    ZnUpscalerStepData,
    ZnUpscalerTensorBoardStage,
    ZnUpscalerTrainStage,
    render_preview_tiled,
    residual_block_mean_loss,
)
from terrain_higan.training import TrainingContext
from terrain_higan.training.losses import residual_aggregate_loss


class ZnUpscalerTrainingCropsTest(unittest.TestCase):
    def test_tiled_preview_respects_z0_export_minimum_at_partial_edges(self):
        generator = Z0Generator((-5.0, 5.0)).eval()
        program = torch.export.export(
            Z0GeneratorExportModule(generator).eval(),
            (torch.randn(2, 1, 64, 64),),
            dynamic_shapes={"noise": {
                0: torch.export.Dim("batch", min=1),
                2: torch.export.Dim("height", min=EXPORT_MINIMUM_INPUT_SIZE),
                3: torch.export.Dim("width", min=EXPORT_MINIMUM_INPUT_SIZE),
            }},
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "generator.pt2"
            torch.export.save(program, path)
            exported_model = torch.export.load(path).module()
            with torch.inference_mode():
                for shape in ((1, 1, 48, 48), (1, 1, 80, 80), (1, 1, 81, 81), (1, 1, 85, 149)):
                    with self.subTest(shape=shape):
                        noise = torch.randn(shape)
                        torch.testing.assert_close(
                            render_preview_tiled(exported_model, noise),
                            generator(noise),
                        )

    def test_tiled_previews_match_full_maps_at_seams_and_partial_tiles(self):
        torch.manual_seed(7)
        z0 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=9),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2, 1, kernel_size=9),
        ).double().eval()
        z0_noise = torch.randn(2, 1, 85, 149, dtype=torch.double)
        z0_tile_shapes = []
        z0.register_forward_hook(lambda module, inputs, output: z0_tile_shapes.append(output.shape[-2:]))
        with torch.inference_mode():
            expected = z0(z0_noise)
            z0_tile_shapes.clear()
            torch.testing.assert_close(render_preview_tiled(z0, z0_noise), expected, rtol=1e-10, atol=1e-12)
        self.assertEqual(len(z0_tile_shapes), 6)
        self.assertTrue(all(height <= 64 and width <= 64 for height, width in z0_tile_shapes))
        for z_level in (1, 20):
            with self.subTest(z_level=z_level):
                model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2)).double().eval()
                input_map = torch.randn(2, 1, 43, 75, dtype=torch.double) * 0.01
                noise = torch.randn(2, 4, 43, 75, dtype=torch.double)
                tile_shapes = []
                model.register_forward_hook(lambda module, inputs, output: tile_shapes.append(output.shape[-2:]))
                with torch.inference_mode():
                    expected = model(input_map, noise)
                    tile_shapes.clear()
                    actual = render_preview_tiled(model, input_map, noise)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                self.assertEqual(len(tile_shapes), 6)
                self.assertTrue(all(height <= 64 and width <= 64 for height, width in tile_shapes))

    def test_selected_context_crops_match_full_map_values_and_gradients(self):
        for z_level, metric_size in ((1, 3), (10, 6), (11, 6), (20, 6)):
            with self.subTest(z_level=z_level), TemporaryDirectory() as directory:
                torch.manual_seed(7)
                root = Path(directory)
                source = root / f"z{z_level - 1}"
                source.mkdir()
                maps = np.random.default_rng(7).normal(0, 0.01, (3, 1, 32, 35)).astype(np.float32)
                np.save(source / "maps_000000.npy", maps)
                halo = 5
                rows = torch.tensor([halo, halo + 3, 32 - metric_size - halo])
                columns = torch.tensor([halo, halo + 5, 35 - metric_size - halo])
                selected = torch.tensor([2, 0])
                stage = MaterializedInputMapStage(
                    maps_dir=root, z_level=z_level, maps_per_batch=3,
                    metric_patch_size=metric_size,
                    input_map_patch_mean_pdf=torch.tensor(((-1.0, 1.0), (1.0, 1.0))),
                    selected_patch_count=2, inverse_pdf_weight_central_fraction=0.95,
                    inverse_pdf_weight_cap_stddevs=5.0, device=torch.device("cpu"),
                )
                data = ZnUpscalerStepData()
                with patch.object(stage, "_sample_metric_coordinates", return_value=(rows, columns)), \
                        patch("terrain_higan.projects.zn_map_upscaler.stages.select_pdf_weighted_values", return_value=selected):
                    stage.run(TrainingContext(step_number=1, step_data=data))

                input_size = metric_size + 2 * halo
                self.assertEqual(data.input_map.shape, (2, 1, input_size, input_size))
                self.assertEqual(data.input_metric_patches.shape, (2, 1, metric_size, metric_size))
                torch.testing.assert_close(data.output_metric_rows, torch.zeros(2, dtype=torch.long))
                torch.testing.assert_close(data.output_metric_columns, torch.zeros(2, dtype=torch.long))
                for crop, metric, index in zip(data.input_map, data.input_metric_patches, selected.tolist()):
                    row, column = rows[index], columns[index]
                    expected = maps[index, :, row-halo:row+metric_size+halo, column-halo:column+metric_size+halo]
                    torch.testing.assert_close(crop, torch.from_numpy(expected), rtol=0, atol=0)
                    torch.testing.assert_close(metric, crop[:, halo:halo+metric_size, halo:halo+metric_size])
                np.testing.assert_array_equal(np.load(source / "maps_000000.npy"), maps)

                model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2)).double()
                full_maps = torch.from_numpy(maps[selected.numpy()]).double()
                full_noise = torch.randn(2, 4, 32, 35, dtype=torch.double)
                cropped_noise = torch.stack([
                    full_noise[index, :, rows[source_index]-halo:rows[source_index]-halo+input_size,
                               columns[source_index]-halo:columns[source_index]-halo+input_size]
                    for index, source_index in enumerate(selected.tolist())
                ])
                full_values = model(full_maps, full_noise)
                crop_values = model(data.input_map.double(), cropped_noise)
                output_size = 2 * metric_size
                cropped_values = center_crop(crop_values, output_size)
                reference_values = torch.stack([
                    full_values[index, :, 2*(rows[source_index]-4):2*(rows[source_index]-4)+output_size,
                           2*(columns[source_index]-4):2*(columns[source_index]-4)+output_size]
                    for index, source_index in enumerate(selected.tolist())
                ])
                torch.testing.assert_close(cropped_values, reference_values, rtol=1e-10, atol=1e-12)
                parameters = tuple(model.parameters())
                full_gradients = torch.autograd.grad(reference_values.square().mean(), parameters)
                crop_gradients = torch.autograd.grad(cropped_values.square().mean(), parameters)
                for cropped, reference in zip(crop_gradients, full_gradients):
                    torch.testing.assert_close(cropped, reference, rtol=1e-10, atol=1e-12)

    def test_block_mean_uses_full_maps_while_other_losses_use_selected_patches(self):
        class RegionModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                values = torch.full((1, 1, 10, 12), 100.0)
                values[..., 2:8, 3:9] = torch.linspace(-0.4, 0.7, 36).reshape(6, 6)
                self.values = torch.nn.Parameter(values)

            def forward(self, input_map, noise):
                residuals = self.values.expand(input_map.shape[0], -1, -1, -1)
                return torch.cat((residuals, residuals * 0.3), dim=1)

        torch.manual_seed(7)
        model = RegionModel()
        gradients = []
        model.values.register_hook(lambda gradient: gradients.append(gradient.detach().clone()))
        critic = ZnUpscalerCritic(z_level=1, layer_specs="4/3 2/1")
        data = ZnUpscalerStepData()
        data.input_map = torch.zeros(2, 1, 11, 11)
        data.input_metric_patches = torch.zeros(2, 1, 3, 3)
        data.output_metric_rows = torch.zeros(2, dtype=torch.long)
        data.output_metric_columns = torch.zeros(2, dtype=torch.long)
        real_pool = torch.linspace(-0.2, 0.2, 64).reshape(4, 1, 4, 4)
        real_pool[1] *= 10
        data.real_patches = real_pool.clone()
        expected_full_residuals = model.values.detach().clone().expand(2, -1, -1, -1)
        expected_residuals = center_crop(model.values.detach(), 6).expand(2, -1, -1, -1)
        expected_residual_patches = model.values.detach()[..., 3:7, 4:8].expand(2, -1, -1, -1)
        stage = ZnUpscalerTrainStage(
            model=model, critic=critic, z_level=1,
            model_optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
            critic_optimizer=torch.optim.SGD(critic.parameters(), lr=0.0),
            patch_mean_pdf=torch.tensor(((-1.0, 1.0), (1.0, 1.0))),
            patch_size=4, patches_per_batch=2,
            residual_aggregate_loss_weight=0.7, residual_bounds_loss_weight=0.4,
            residual_block_mean_loss_weight=0.3, residual_limits=(-0.1, 0.1),
            inverse_pdf_weight_central_fraction=0.95, inverse_pdf_weight_cap_stddevs=5.0,
        )
        with patch("torch.randint", return_value=torch.ones(2, dtype=torch.long)), \
                patch("terrain_higan.projects.zn_map_upscaler.stages.select_pdf_weighted_patches", return_value=torch.tensor([2, 0])):
            stage.run(TrainingContext(step_number=1, step_data=data))

        self.assertEqual(data.output_map.shape, (2, 1, 6, 6))
        self.assertEqual(data.generated_patches.shape, (2, 1, 4, 4))
        torch.testing.assert_close(data.denormalized_residuals, expected_residuals)
        torch.testing.assert_close(data.denormalized_residual_patches, expected_residual_patches)
        torch.testing.assert_close(data.generated_patches, expected_residual_patches * 0.3)
        expected_bounds = (
            F.relu(-0.1 - expected_residual_patches).square() + F.relu(expected_residual_patches - 0.1).square()
        ).mean() / UPSCALE_STATS[1].residual_divisor ** 2
        torch.testing.assert_close(data.generator_residual_bounds_loss, expected_bounds)
        selected_real = real_pool[torch.tensor([2, 0])]
        torch.testing.assert_close(data.real_patches, selected_real)
        expected_real_residuals = selected_real - F.interpolate(F.avg_pool2d(selected_real, 2), scale_factor=2, mode="nearest")
        torch.testing.assert_close(
            data.generator_residual_aggregate_loss,
            residual_aggregate_loss(normalize_residuals(expected_residual_patches, 1), normalize_residuals(expected_real_residuals, 1)),
        )
        torch.testing.assert_close(
            data.generator_residual_block_mean_loss,
            residual_block_mean_loss(expected_full_residuals, UPSCALE_STATS[1].residual_divisor),
        )
        torch.testing.assert_close(
            data.generator_loss,
            F.elu(-data.fake_scores).mean() + 0.7 * data.generator_residual_aggregate_loss
            + 0.4 * expected_bounds + 0.3 * data.generator_residual_block_mean_loss,
        )
        self.assertEqual(len(gradients), 1)
        gradient = gradients[0].clone()
        self.assertGreater(gradient[..., 3:7, 4:8].abs().sum().item(), 0)
        gradient[..., 3:7, 4:8] = 0
        expected_gradient = torch.ones_like(gradient) * (
            0.3 * 2 * expected_full_residuals.mean()
            / (UPSCALE_STATS[1].residual_divisor ** 2 * model.values.numel())
        )
        expected_gradient[..., 3:7, 4:8] = 0
        torch.testing.assert_close(gradient, expected_gradient)

    def test_bilinear_crop_diagnostics_align_with_the_metric_parent(self):
        torch.manual_seed(7)
        model = ZnUpscalerModel(z_level=11, layer_channels=(8, 8, 4, 2))
        data = ZnUpscalerStepData()
        data.input_map = torch.randn(2, 1, 16, 16) * 0.01
        data.input_metric_patches = center_crop(data.input_map, 6)
        residuals, output = model(data.input_map, torch.randn(2, 4, 16, 16)).split(1, dim=1)
        data.denormalized_residuals = center_crop(residuals, 12).detach()
        data.output_map = center_crop(output, 12).detach()
        data.real_residual_block_means = torch.zeros(2, 1, 5, 5)
        for name in (
            "generator_loss", "generator_adversarial_loss", "generator_residual_aggregate_loss",
            "generator_residual_bounds_loss", "generator_residual_block_mean_loss",
            "generator_residual_block_mean_loss_weighted", "critic_loss",
        ):
            setattr(data, name, torch.tensor(0.0))
        writer = Mock()
        logger = ZnUpscalerTensorBoardStage(writer, z_level=11, log_every_steps=1, image_every_steps=10)
        data.input_map = data.input_map + 0.1
        logger._log_scalars(data, 1)
        scalars = {call.args[0]: call.args[1] for call in writer.add_scalar.call_args_list}
        drift = (F.avg_pool2d(data.output_map, 2, 2) - data.input_metric_patches) * ELEVATION_STD_M
        self.assertEqual(scalars["consistency/output_parent_drift_meters_mean"], drift.mean().item())
        self.assertEqual(scalars["consistency/output_parent_drift_meters_rms"], drift.square().mean().sqrt().item())


if __name__ == "__main__":
    unittest.main()

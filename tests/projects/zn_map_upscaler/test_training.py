import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from terrain_higan.etopo.normalization import UPSCALE_STATS, normalize_residuals
from terrain_higan.projects.zn_map_upscaler import stages
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.generate_input_maps import MapBatchSource
from terrain_higan.projects.zn_map_upscaler.geometry import center_crop
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerCritic, ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.save import save_zn_map_upscaler
from terrain_higan.projects.zn_map_upscaler.train import _probe_model_save
from terrain_higan.training import TrainingContext
from terrain_higan.training.losses import residual_aggregate_loss

SCALAR_TAGS = [
    "loss/generator", "loss/generator_adversarial", "loss/generator_residual_aggregate",
    "loss/generator_residual_bounds",
    "loss/generator_residual_block_mean",
    "loss/generator_residual_block_mean_weighted", "loss/critic",
    "residuals/block_mean_meters_mean", "residuals/block_mean_meters_rms",
    "consistency/output_parent_drift_meters_mean", "consistency/output_parent_drift_meters_rms",
]
HISTOGRAM_TAGS = ["residuals/denormalized", "scores/critic_real", "scores/critic_fake"]
IMAGE_TAGS = [
    "images/critic_real_highest", "images/critic_real_lowest",
    "images/critic_fake_highest", "images/critic_fake_lowest",
    "images/input_maps", "images/input_metric_patches", "images/output_maps",
    "images/generated_patches", "images/output_map_high_pass_features",
    "images/output_map_low_pass_features", "images/input_map_high_pass_features",
    "images/input_map_low_pass_features", "images/real_patches",
    "images/real_patches_normalized", "images/generated_patches_normalized",
]


class ZnUpscalerTrainingTest(unittest.TestCase):
    def test_bounds_loss_is_invariant_to_batch_patch_size_and_zoom(self) -> None:
        for z_level in (1, 20):
            for repeats in ((1, 1, 1, 1), (3, 1, 2, 2)):
                with self.subTest(z_level=z_level, repeats=repeats):
                    divisor = UPSCALE_STATS[z_level].residual_divisor
                    normalized = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], requires_grad=True)
                    residuals = normalized.reshape(1, 1, 1, 5).repeat(*repeats) * divisor
                    loss = stages.residual_bounds_loss(residuals, (-2 * divisor, 3 * divisor), divisor)
                    torch.testing.assert_close(loss, torch.tensor(1.0))
                    loss.backward()
                    torch.testing.assert_close(normalized.grad, torch.tensor([-0.8, 0.0, 0.0, 0.0, 0.4]))

    def test_invalid_materialized_maps_fail_before_coordinate_sampling(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "z0"
            source.mkdir()
            path = source / "maps_000000.npy"
            np.save(path, np.zeros((2, 1, 12, 12), dtype=np.float32))
            stage = stages.MaterializedInputMapStage(
                maps_dir=root, z_level=1, maps_per_batch=2, metric_patch_size=3,
                input_map_patch_mean_pdf=torch.tensor(((-1.0, 1.0), (1.0, 1.0))),
                selected_patch_count=1, inverse_pdf_weight_central_fraction=0.95,
                inverse_pdf_weight_cap_stddevs=5.0, device=torch.device("cpu"),
            )
            for shape, message in (((2, 2, 12, 12), "B1HW"), ((2, 1, 10, 12), "margin")):
                with self.subTest(shape=shape):
                    np.save(path, np.zeros(shape, dtype=np.float32))
                    with patch.object(stage, "_sample_metric_coordinates") as sample:
                        with self.assertRaisesRegex(ValueError, message):
                            stage.run(TrainingContext(step_number=1, step_data=stages.ZnUpscalerStepData()))
                        sample.assert_not_called()

    def test_preview_restores_live_model_mode_on_failure_without_evaluating_export(self) -> None:
        class ExportedZ0(torch.nn.Module):
            def eval(self):
                raise AssertionError("exported modules must not receive eval()")

            def forward(self, noise):
                return noise[..., 8:-8, 8:-8]

        class FailingUpscaler(torch.nn.Module):
            def forward(self, input_map, noise):
                raise RuntimeError("test inference failure")

        writer = Mock()
        model = FailingUpscaler().train()
        with patch("torch.export.load", return_value=SimpleNamespace(module=ExportedZ0)):
            stage = stages.ZnUpscalerLargeMapPreviewStage(
                writer, z_level=1, model=model, z0_pt2_path=Path("unused.pt2"),
                predecessor_model_paths=(), image_every_steps=3,
                device=torch.device("cpu"), target_size=32,
            )
        for was_training in (True, False):
            with self.subTest(was_training=was_training):
                model.train(was_training)
                with self.assertRaisesRegex(RuntimeError, "test inference failure"):
                    stage.run(TrainingContext(step_number=3, step_data=stages.ZnUpscalerStepData()))
                self.assertEqual(model.training, was_training)
        writer.add_image.assert_not_called()

    def test_center_crop_is_a_view_and_handles_odd_rectangular_maps(self) -> None:
        maps = torch.arange(7 * 9).reshape(1, 1, 7, 9)
        cropped = center_crop(maps, 4)
        torch.testing.assert_close(cropped, maps[..., 1:5, 2:6], rtol=0, atol=0)
        cropped[0, 0, 0, 0] = -1
        self.assertEqual(maps[0, 0, 1, 2].item(), -1)
        with self.assertRaisesRegex(ValueError, "Cannot crop"):
            center_crop(maps, 8)

    def test_materialized_source_returns_independent_writable_batches(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            for index, values in enumerate(([0, 1], [2, 3, 4])):
                maps = np.array(values, dtype=np.float32)[:, None, None, None]
                np.save(path / f"maps_{index:06d}.npy", np.broadcast_to(maps, (len(values), 1, 2, 2)))
            source = MapBatchSource(path, map_size=2, start_sample=1)
            for count, expected in ((1, [1]), (5, [2, 3, 4, 0, 1]), (2, [2, 3])):
                batch = source.take(count)
                self.assertEqual(batch[:, 0, 0, 0].tolist(), expected)
                self.assertEqual(batch.dtype, np.float32)
                self.assertTrue(batch.flags.writeable)
                batch.fill(-1)
            self.assertEqual(np.load(path / "maps_000000.npy")[:, 0, 0, 0].tolist(), [0, 1])
            self.assertEqual(np.load(path / "maps_000001.npy")[:, 0, 0, 0].tolist(), [2, 3, 4])

    def test_region_sampler_preserves_coordinates_rng_and_gradients(self) -> None:
        maps = torch.arange(2 * 2 * 12 * 14, dtype=torch.float32).reshape(2, 2, 12, 14)
        maps.requires_grad_()
        rows, columns = torch.tensor([0, 6]), torch.tensor([1, 8])
        torch.manual_seed(17)
        output_rows = rows + torch.randint(3, rows.shape)
        output_columns = columns + torch.randint(3, columns.shape)
        expected_rng = torch.get_rng_state()
        expected = torch.stack([
            maps[index, :, row:row + 4, column:column + 4]
            for index, (row, column) in enumerate(zip(output_rows, output_columns))
        ])
        expected_gradient = torch.autograd.grad(expected.sum(), maps)[0]
        torch.manual_seed(17)
        with patch.object(stages, "_extract_patches", wraps=stages._extract_patches) as extract:
            patches = stages.sample_patches_from_regions(
                maps, rows=rows, columns=columns, region_size=6, patch_size=4,
            )
        self.assertEqual(extract.call_count, 1)
        torch.testing.assert_close(patches, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(patches.sum(), maps)[0], expected_gradient)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for bad_rows, region_size, message in (
                (rows[:1], 6, "one output-space coordinate"),
                (rows, 3, "Cannot sample"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                stages.sample_patches_from_regions(
                    maps, rows=bad_rows, columns=columns, region_size=region_size, patch_size=4,
                )

    def test_cpu_training(self) -> None:
        for z_level in (1, 3, 4, 5, 11, 20):
            with self.subTest(z_level=z_level):
                _check_training(z_level, "cpu")

    def test_block_mean_loss_can_be_disabled(self) -> None:
        _, data = _check_training(20, "cpu", block_mean_loss_weight=0.0)
        self.assertGreater(data.generator_residual_block_mean_loss.item(), 0.0)
        self.assertEqual(data.generator_residual_block_mean_loss_weighted.item(), 0.0)

    @pytest.mark.cuda
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_training(self) -> None:
        for z_level in (1, 11, 20):
            with self.subTest(z_level=z_level):
                _check_training(z_level, "cuda")

    def test_logging_intervals_tags_and_event_file(self) -> None:
        _, data = _check_training(1, "cpu")
        writer = Mock()
        stage = stages.ZnUpscalerTensorBoardStage(writer, z_level=1, log_every_steps=2, image_every_steps=3)
        rng = torch.get_rng_state()
        for step in range(1, 7):
            stage.run(TrainingContext(step_number=step, step_data=data))
        self.assertEqual([call.args[0] for call in writer.add_scalar.call_args_list], SCALAR_TAGS * 3)
        self.assertEqual([call.args[0] for call in writer.add_histogram.call_args_list], HISTOGRAM_TAGS * 3)
        self.assertEqual([call.args[0] for call in writer.add_image.call_args_list], IMAGE_TAGS * 2)
        image_count = len(IMAGE_TAGS)
        self.assertEqual([call.args[2] for call in writer.add_image.call_args_list], [3] * image_count + [6] * image_count)
        for first, second in zip(writer.add_image.call_args_list[:image_count], writer.add_image.call_args_list[image_count:]):
            torch.testing.assert_close(first.args[1], second.args[1], rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        writer.flush.assert_not_called()
        with TemporaryDirectory() as directory:
            with SummaryWriter(directory) as real_writer:
                stages.ZnUpscalerTensorBoardStage(
                    real_writer, z_level=1, log_every_steps=1, image_every_steps=1,
                ).run(TrainingContext(step_number=1, step_data=data))
            tags = EventAccumulator(directory).Reload().Tags()
            self.assertEqual(set(tags["scalars"]), set(SCALAR_TAGS))
            self.assertEqual(set(tags["histograms"]), set(HISTOGRAM_TAGS))
            self.assertEqual(set(tags["images"]), set(IMAGE_TAGS))

    def test_critic_extremes_rank_patches_by_mean_spatial_score(self) -> None:
        data = stages.ZnUpscalerStepData()
        data.real_patches = torch.arange(3.0).reshape(3, 1, 1, 1)
        data.generated_patches = data.real_patches + 10
        data.real_scores = torch.tensor([[10.0, -10.0], [2.0, 2.0], [1.0, 1.0]])
        data.fake_scores = -data.real_scores
        writer = Mock()
        logger = stages.ZnUpscalerTensorBoardStage(
            writer, z_level=1, log_every_steps=1, image_every_steps=1,
            critic_extreme_patch_count=2,
        )
        with patch.object(stages, "make_scored_patch_grid", return_value=torch.zeros(3, 1, 1)) as grid:
            logger._log_critic_extremes(data, 1)
        expected_indices = ([1, 2], [0, 2], [0, 2], [1, 2])
        for call, indices, patches, scores in zip(
            grid.call_args_list, expected_indices,
            (data.real_patches, data.real_patches, data.generated_patches, data.generated_patches),
            (data.real_scores, data.real_scores, data.fake_scores, data.fake_scores),
        ):
            torch.testing.assert_close(call.args[0], patches[indices])
            torch.testing.assert_close(call.args[1], scores.mean(dim=1)[indices])

    def test_export_failure_cleans_temporary_artifacts_without_publishing(self) -> None:
        model = ZnUpscalerModel(z_level=1, layer_channels=(8, 8, 4, 2))
        with TemporaryDirectory() as directory:
            config = ZnUpscalerConfig(model_dir=Path(directory))
            onnx = Mock()

            def fail_save(path):
                Path(path).write_bytes(b"partial")
                raise RuntimeError("test export failure")

            onnx.save.side_effect = fail_save
            with patch("torch.export.export", return_value=Mock()), patch("torch.onnx.export", return_value=onnx), \
                    patch("torch.export.save", side_effect=lambda program, path: Path(path).write_bytes(b"pt2")):
                with self.assertRaisesRegex(RuntimeError, "test export failure"):
                    save_zn_map_upscaler(model, config)
            self.assertEqual(list(config.model_run_dir.iterdir()), [])
            self.assertFalse((config.model_dir / "z1/latest").exists())
            self.assertTrue(model.training)

    def test_startup_probe_checks_all_artifacts_and_reloads_explicit_configuration(self) -> None:
        config = ZnUpscalerConfig(z_level=11)
        model = Mock()

        def fake_save(model_arg, config_arg, *, output_dir, update_latest):
            self.assertIs(model_arg, model)
            self.assertIs(config_arg, config)
            self.assertFalse(update_latest)
            paths = tuple(output_dir / filename for filename in ("upscaler.safetensors", "upscaler.pt2", "upscaler.onnx"))
            for path in paths:
                path.write_bytes(b"artifact")
            return paths

        with patch("terrain_higan.projects.zn_map_upscaler.train.save_zn_map_upscaler", side_effect=fake_save), \
                patch("terrain_higan.projects.zn_map_upscaler.train.load_zn_map_upscaler") as load:
            _probe_model_save(model, config)
            self.assertEqual(load.call_args.kwargs, {"z_level": 11, "layer_channels": config.model_layer_channels})
            self.assertFalse(load.call_args.args[0].exists())

        def empty_save(*args, **kwargs):
            paths = fake_save(*args, **kwargs)
            paths[-1].write_bytes(b"")
            return paths

        with patch("terrain_higan.projects.zn_map_upscaler.train.save_zn_map_upscaler", side_effect=empty_save), \
                patch("terrain_higan.projects.zn_map_upscaler.train.load_zn_map_upscaler") as load:
            with self.assertRaisesRegex(RuntimeError, "did not write"):
                _probe_model_save(model, config)
            load.assert_not_called()


def _check_training(
        z_level: int, device_name: str, block_mean_loss_weight: float = 0.3,
) -> tuple[stages.ZnUpscalerTrainStage, stages.ZnUpscalerStepData]:
    torch.manual_seed(7)
    device = torch.device(device_name)
    model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2)).to(device)
    critic = ZnUpscalerCritic(z_level=z_level, layer_specs="4/3 2/1").to(device)
    initial_model = [parameter.detach().clone() for parameter in model.parameters()]
    initial_critic = [parameter.detach().clone() for parameter in critic.parameters()]
    data = stages.ZnUpscalerStepData()
    data.real_patches = torch.linspace(-0.1, 0.1, 400, device=device).reshape(4, 1, 10, 10)
    input_size = 16 if z_level > 10 else 14
    data.input_map = torch.zeros(2, 1, input_size, input_size, device=device)
    data.input_metric_patches = torch.zeros(2, 1, 6, 6, device=device)
    data.output_metric_rows = torch.zeros(2, dtype=torch.long, device=device)
    data.output_metric_columns = torch.zeros(2, dtype=torch.long, device=device)
    context = TrainingContext(step_number=1, step_data=data)
    if z_level > 10:
        stages.ScaleRealPatchesStage(z_level).run(context)
    source = data.real_source_patches if z_level > 10 else data.real_patches
    expected_real = normalize_residuals(
        data.real_patches - F.interpolate(F.avg_pool2d(source, 2), scale_factor=2, mode="nearest"), z_level,
    )
    bounds = (-0.001, 0.001)
    config = ZnUpscalerConfig(z_level=z_level, generator_residual_block_mean_loss_weight=block_mean_loss_weight)
    effective_block_mean_weight = (
        config.generator_residual_block_mean_loss_weight * config.level_config.residual_block_mean_loss_multiplier
    )
    stage = stages.ZnUpscalerTrainStage(
        model=model, critic=critic, z_level=z_level,
        model_optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.0, 0.9)),
        critic_optimizer=torch.optim.AdamW(critic.parameters(), lr=1e-4, betas=(0.0, 0.9)),
        patch_mean_pdf=torch.tensor(((-1.0, 1.0), (1.0, 1.0)), device=device),
        patch_size=10, patches_per_batch=2,
        residual_aggregate_loss_weight=0.7, residual_bounds_loss_weight=0.4,
        residual_block_mean_loss_weight=effective_block_mean_weight,
        residual_limits=bounds, inverse_pdf_weight_central_fraction=0.95,
        inverse_pdf_weight_cap_stddevs=5.0,
    )
    real_indices = torch.tensor([2, 0], device=device)
    full_residuals = []
    hook = model.register_forward_hook(lambda module, inputs, output: full_residuals.append(output[:, :1].detach().clone()))
    with patch.object(stages, "select_pdf_weighted_patches", return_value=real_indices):
        stage.run(context)
    hook.remove()
    residual_patches = data.denormalized_residual_patches
    expected_aggregate = residual_aggregate_loss(normalize_residuals(residual_patches, z_level), expected_real[real_indices])
    expected_bounds = (
        F.relu(bounds[0] - residual_patches).square() + F.relu(residual_patches - bounds[1]).square()
    ).mean() / UPSCALE_STATS[z_level].residual_divisor ** 2
    torch.testing.assert_close(data.generator_residual_aggregate_loss, expected_aggregate)
    torch.testing.assert_close(data.generator_residual_bounds_loss, expected_bounds)
    expected_block_mean = (
        full_residuals[0].mean(dim=(-2, -1)) / UPSCALE_STATS[z_level].residual_divisor
    ).square().mean()
    torch.testing.assert_close(data.generator_residual_block_mean_loss, expected_block_mean)
    torch.testing.assert_close(data.generator_residual_block_mean_loss_weighted, effective_block_mean_weight * expected_block_mean)
    torch.testing.assert_close(
        data.generator_loss,
        data.generator_adversarial_loss + 0.7 * expected_aggregate + 0.4 * expected_bounds
        + effective_block_mean_weight * expected_block_mean,
    )
    assert not data.generator_residual_block_mean_loss.requires_grad
    assert not data.generator_residual_block_mean_loss_weighted.requires_grad
    torch.testing.assert_close(data.critic_loss, F.elu(data.fake_scores).mean() + F.elu(-data.real_scores).mean())
    assert data.real_patches.shape == data.generated_patches.shape == (2, 1, 10, 10)
    for module, initial in ((model, initial_model), (critic, initial_critic)):
        assert any(not torch.equal(old, new) for old, new in zip(initial, module.parameters()))
        assert all(torch.isfinite(parameter).all() and parameter.grad is None for parameter in module.parameters())
    assert not data.output_map.requires_grad
    return stage, data

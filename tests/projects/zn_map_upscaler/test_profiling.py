import json
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch
import pytest
from torch.utils.tensorboard import SummaryWriter

from terrain_higan.projects.zn_map_upscaler import config as config_module
from terrain_higan.projects.zn_map_upscaler import profiling
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerCritic, ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.stages import (
    MaterializedInputMapStage,
    ZnUpscalerLargeMapPreviewStage,
    ZnUpscalerStepData,
    ZnUpscalerTensorBoardStage,
    ZnUpscalerTrainStage,
)
from terrain_higan.projects.zn_map_upscaler.train import train_zn_map_upscaler
from terrain_higan.training import PatchLoaderStage, TrainingLoop


class CountingStage:
    def __init__(self, fail_at=None):
        self.steps = []
        self.fail_at = fail_at

    def run(self, context):
        self.steps.append(context.step_number)
        with profiling.profile_range("test/work"):
            context.step_data.value = torch.arange(16).square().sum()
        if context.step_number == self.fail_at:
            raise RuntimeError("stage failure")


def profiling_config(root, **kwargs):
    options = dict(
        device="cpu", profile=True, profile_wait_steps=2,
        profile_warmup_steps=1, profile_active_steps=2,
        minimum_steps=0,
    )
    options.update(kwargs)
    config = ZnUpscalerConfig(**options)
    object.__setattr__(config, "run_dir", root)
    return config


class ZnUpscalerProfilingTest(unittest.TestCase):
    def test_configuration_rejects_invalid_windows(self):
        for kwargs in (
            {"input_map_loading_workers": 0},
            {"profile_only": True},
            {"profile": True, "profile_wait_steps": -1},
            {"profile": True, "profile_warmup_steps": -1},
            {"profile": True, "profile_active_steps": 0},
            {"profile": True, "num_steps": 32},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ZnUpscalerConfig(**kwargs)

    def test_disabled_profiling_runs_normally_without_recording(self):
        stage = CountingStage()
        loop = TrainingLoop([stage], ZnUpscalerStepData, 3)
        with patch.object(torch.profiler, "profile") as profiler, \
                patch.object(torch.profiler, "record_function") as record:
            profiling.run_training_loop(loop, ZnUpscalerConfig(num_steps=3, minimum_steps=0))
        self.assertEqual(stage.steps, [1, 2, 3])
        profiler.assert_not_called()
        record.assert_not_called()

    def test_capture_has_exact_steps_and_training_can_continue(self):
        for profile_only, num_steps, expected_steps in ((True, 8, 5), (True, 5, 5), (False, 8, 8)):
            with self.subTest(profile_only=profile_only, num_steps=num_steps), TemporaryDirectory() as directory:
                root = Path(directory)
                config = profiling_config(
                    root, num_steps=num_steps, profile_only=profile_only,
                    profile_shapes=True, profile_memory=True, profile_stacks=True,
                )
                stage = CountingStage()
                loop = TrainingLoop([profiling.ProfiledStage(stage)], ZnUpscalerStepData, num_steps)
                profiling.run_training_loop(loop, config)
                self.assertEqual(stage.steps, list(range(1, expected_steps + 1)))
                output = root / "profiling"
                metadata = json.loads(next(output.glob("*.metadata.json")).read_text())
                self.assertEqual((metadata["active_first_step"], metadata["active_last_step"]), (4, 5))
                self.assertTrue(metadata["complete"])
                self.assertGreater(metadata["window_timings"]["active_seconds"], 0)
                self.assertGreater(metadata["profile_to_baseline_time_ratio"], 0)
                self.assertIsNone(metadata["gpu_name"])
                ranges = json.loads(next(output.glob("*.ranges.json")).read_text())["ranges"]
                self.assertEqual(next(row["calls"] for row in ranges if row["name"] == "zn/test/work"), 2)
                trace = json.loads(next(output.glob("*.trace.json")).read_text())
                steps = [event["name"] for event in trace["traceEvents"] if event.get("name", "").startswith("ProfilerStep#")]
                self.assertEqual(steps, ["ProfilerStep#3", "ProfilerStep#4"])
                self.assertIn("Input Shapes", next(output.glob("*.cpu.txt")).read_text())
                self.assertTrue(next(output.glob("*.device.txt")).stat().st_size)
                self.assertFalse(profiling._RANGES_ENABLED.get())

    def test_exception_flushes_partial_trace_and_restores_ranges(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = profiling_config(root, num_steps=8, profile_only=True)
            stage = CountingStage(fail_at=4)
            loop = TrainingLoop([profiling.ProfiledStage(stage)], ZnUpscalerStepData, 8)
            with self.assertRaisesRegex(RuntimeError, "stage failure"):
                profiling.run_training_loop(loop, config)
            self.assertFalse(profiling._RANGES_ENABLED.get())
            metadata = json.loads(next((root / "profiling").glob("*.metadata.json")).read_text())
            self.assertFalse(metadata["complete"])

    def test_zero_wait_and_warmup_windows(self):
        for wait, warmup in ((0, 0), (0, 1), (1, 0)):
            with self.subTest(wait=wait, warmup=warmup), TemporaryDirectory() as directory:
                config = profiling_config(Path(directory), profile_wait_steps=wait,
                                          profile_warmup_steps=warmup, profile_active_steps=1,
                                          num_steps=wait + warmup + 1, profile_only=True)
                stage = CountingStage()
                loop = TrainingLoop([profiling.ProfiledStage(stage)], ZnUpscalerStepData, config.num_steps)
                profiling.run_training_loop(loop, config)
                metadata = json.loads(next((config.run_dir / "profiling").glob("*.metadata.json")).read_text())
                self.assertEqual(metadata["active_first_step"], config.num_steps)
                self.assertEqual(metadata["active_last_step"], config.num_steps)
                ranges = json.loads(next((config.run_dir / "profiling").glob("*.ranges.json")).read_text())["ranges"]
                self.assertEqual(next(row["calls"] for row in ranges if row["name"] == "zn/test/work"), 1)

    def test_default_logging_intervals_land_in_requested_windows(self):
        for wait, expect_images in ((20, False), (90, False), (2490, True)):
            with self.subTest(wait=wait), TemporaryDirectory() as directory:
                config = profiling_config(Path(directory), profile_wait_steps=wait,
                                          profile_active_steps=12, profile_only=True)
                stage = CountingStage()
                logging = ZnUpscalerTensorBoardStage(Mock(), z_level=1, log_every_steps=100, image_every_steps=2500)
                logging._log_scalars = Mock()
                logging._log_histograms = Mock()
                logging._log_images = Mock()
                loop = TrainingLoop([profiling.ProfiledStage(stage), profiling.ProfiledStage(logging)], ZnUpscalerStepData, config.num_steps)
                profiling.run_training_loop(loop, config)
                names = {row["name"] for row in json.loads(next((config.run_dir / "profiling").glob("*.ranges.json")).read_text())["ranges"]}
                self.assertEqual("zn/logging/scalars" in names, wait != 20)
                self.assertEqual("zn/logging/images" in names, expect_images)

    def test_missing_cuda_kernel_trace_is_reported_and_preserved(self):
        with TemporaryDirectory() as directory:
            config = profiling_config(Path(directory))
            config = replace(config, device="cuda")
            object.__setattr__(config, "run_dir", Path(directory))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                torch.arange(4).sum()
            with patch.object(torch.cuda, "get_device_name", return_value="test GPU"):
                with self.assertRaisesRegex(RuntimeError, "No CUDA kernels"):
                    profiling._save_capture(profiler, config, config.profile_total_steps, {})
            self.assertTrue(next((config.run_dir / "profiling").glob("*.trace.json")).stat().st_size)

    def test_cuda_unavailable_is_reported(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA profiling is unavailable"):
                profiling.validate_profile_device(ZnUpscalerConfig(profile=True))

    def test_profile_only_skips_durable_artifacts_and_closes_writer(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            config = profiling_config(
                root / "run", profile_only=True, analysis_dir=root,
                input_maps_per_batch=32, patches_per_batch=3,
            )
            config.patch_mean_pdf_path.touch()
            shard_dir = root / "z1" / "4x4"
            shard_dir.mkdir(parents=True)
            (shard_dir / "shard-000").touch()
            prefix = "terrain_higan.projects.zn_map_upscaler.train."
            stack.enter_context(patch(prefix + "DEFAULT_ETOPO_BASE_DIR", root))
            stack.enter_context(patch(prefix + "load_patch_mean_pdf", return_value=torch.tensor(((-1.0, 1.0), (1.0, 1.0)))))
            mocked_stages = {}
            for name in (
                "analyze_training_input_maps", "_probe_model_save", "MaterializedInputMapStage",
                "PatchLoaderStage", "ZnUpscalerTrainStage", "ZnUpscalerTensorBoardStage",
                "ZnUpscalerLargeMapPreviewStage", "run_training_loop",
            ):
                mocked_stages[name] = stack.enter_context(patch(prefix + name))
            writer = stack.enter_context(patch(prefix + "SummaryWriter")).return_value
            save = stack.enter_context(patch(prefix + "save_zn_map_upscaler"))
            materialize = stack.enter_context(patch(prefix + "materialize_trained_maps"))
            config.run_dir.mkdir()
            train_zn_map_upscaler(config)
            map_settings = mocked_stages["MaterializedInputMapStage"].call_args.kwargs
            real_settings = mocked_stages["PatchLoaderStage"].call_args.kwargs
            train_settings = mocked_stages["ZnUpscalerTrainStage"].call_args.kwargs
            self.assertEqual(map_settings["maps_per_batch"], config.input_maps_per_batch)
            self.assertEqual(real_settings["patches_per_batch"], config.input_maps_per_batch)
            self.assertEqual(map_settings["selected_patch_count"], config.patches_per_batch)
            self.assertEqual(train_settings["patches_per_batch"], config.patches_per_batch)
            save.assert_not_called()
            materialize.assert_not_called()
            writer.close.assert_called_once()

    @pytest.mark.cuda
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA host required")
    def test_cuda_capture_includes_training_prefetch_and_periodic_logging(self):
        with TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            level = replace(config_module.Z_LEVEL_CONFIGS[0], model_layer_channels=(8, 8, 4, 2))
            stack.enter_context(patch.object(config_module, "Z_LEVEL_CONFIGS", [level]))
            config = profiling_config(root / "run", profile_only=True, num_steps=5,
                                      input_maps_per_batch=32, patches_per_batch=4, input_map_size=16,
                                      log_every_steps=4, image_every_steps=5)
            config = replace(config, device="cuda")
            object.__setattr__(config, "run_dir", root / "run")
            device = torch.device("cuda")
            source = root / "maps" / "z0"
            source.mkdir(parents=True)
            np.save(source / "maps_000000.npy", np.random.default_rng(7).normal(0, 0.01, (32, 1, 16, 16)).astype(np.float32))
            shards = root / "shards" / "z1" / "4x4"
            shards.mkdir(parents=True)
            (shards / "shard-000").write_bytes(np.linspace(-100, 100, 32 * 16, dtype=np.float32).tobytes())
            pdf = torch.tensor(((-1.0, 1.0), (1.0, 1.0)), device=device)
            model = ZnUpscalerModel(z_level=1, layer_channels=config.model_layer_channels).to(device)
            critic = ZnUpscalerCritic(z_level=1, layer_specs=config.critic_layer_specs).to(device)
            writer = stack.enter_context(SummaryWriter(config.run_dir))
            z0_model = Mock()
            z0_model.to.return_value = lambda noise: noise[..., 8:-8, 8:-8]
            stack.enter_context(patch("torch.export.load", return_value=Mock(module=lambda: z0_model)))
            stages = [
                MaterializedInputMapStage(
                    maps_dir=root / "maps", z_level=1, maps_per_batch=32,
                    metric_patch_size=3, input_map_patch_mean_pdf=pdf, selected_patch_count=4,
                    inverse_pdf_weight_central_fraction=0.95, inverse_pdf_weight_cap_stddevs=5.0, device=device,
                ),
                PatchLoaderStage(base_dir=root / "shards", z_level=1, patch_size=4, patches_per_batch=32, device=device),
                ZnUpscalerTrainStage(
                    model=model, critic=critic, z_level=1,
                    model_optimizer=torch.optim.AdamW(model.parameters()), critic_optimizer=torch.optim.AdamW(critic.parameters()),
                    patch_mean_pdf=pdf, patch_size=4, patches_per_batch=4,
                    residual_aggregate_loss_weight=1.0, residual_bounds_loss_weight=1.0,
                    residual_block_mean_loss_weight=0.01, residual_limits=config.residual_limits,
                    inverse_pdf_weight_central_fraction=0.95, inverse_pdf_weight_cap_stddevs=5.0,
                ),
                ZnUpscalerTensorBoardStage(writer, z_level=1, log_every_steps=4, image_every_steps=5),
                ZnUpscalerLargeMapPreviewStage(writer, z_level=1, model=model, z0_pt2_path=Path("unused.pt2"),
                                              predecessor_model_paths=(), image_every_steps=5, device=device, target_size=32),
            ]
            stack.callback(stages[0].close)
            profiling.run_training_loop(TrainingLoop([profiling.ProfiledStage(stage) for stage in stages], ZnUpscalerStepData, 5), config)
            output = config.run_dir / "profiling"
            metadata = json.loads(next(output.glob("*.metadata.json")).read_text())
            self.assertGreater(metadata["cuda_kernel_events"], 0)
            ranges = {row["name"]: row for row in json.loads(next(output.glob("*.ranges.json")).read_text())["ranges"]}
            for name in ("train/generator_gradients", "train/critic_gradients", "logging/scalars", "logging/histograms", "logging/images", "preview/inference"):
                self.assertIn("zn/" + name, ranges)
                self.assertGreater(ranges["zn/" + name]["cpu_total_us"], 0)
            self.assertNotIn("zn/", next(output.glob("*.device.txt")).read_text())
            trace = json.loads(next(output.glob("*.trace.json")).read_text())
            copies = [event for event in trace["traceEvents"] if event.get("cat") == "gpu_memcpy"]
            self.assertTrue(copies)
            self.assertGreater(len({event["args"]["stream"] for event in copies}), 1)


if __name__ == "__main__":
    unittest.main()

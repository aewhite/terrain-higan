from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import weakref

import numpy as np
import torch

from terrain_higan.projects.zn_map_upscaler import train
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.generate_input_maps import (
    MapBatchSource,
    _prepare_output_dir,
    generate_maps,
)
from terrain_higan.projects.zn_map_upscaler.geometry import center_crop
from terrain_higan.projects.zn_map_upscaler.analyze_input_maps import _sample_patch_means


class MapMaterializationTest(unittest.TestCase):
    def test_materialization_splits_2048_maps_into_four_512_map_calls(self) -> None:
        observed_batch_sizes = []

        class RecordingUpscaler(torch.nn.Module):
            def forward(self, parents, noise):
                observed_batch_sizes.append(parents.shape[0])
                assert parents.shape[0] == noise.shape[0]
                return torch.nn.functional.interpolate(parents, scale_factor=2, mode="nearest")[..., 8:-8, 8:-8]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "maps/z0"
            source_dir.mkdir(parents=True)
            np.save(source_dir / "maps_000000.npy", np.full((1, 1, 32, 32), 7, dtype=np.float32))
            config = ZnUpscalerConfig(
                maps_dir=root / "maps", model_dir=root / "models",
                input_map_total_samples=2048, input_maps_per_batch=2048,
                input_map_generation_batch_size=512,
            )
            with patch("terrain_higan.projects.zn_map_upscaler.generate_input_maps.load_zn_map_upscaler", return_value=RecordingUpscaler()):
                train.materialize_trained_maps(config, torch.device("cpu"))
            self.assertEqual(observed_batch_sizes, [512, 512, 512, 512])
            files = list(config.output_map_dir.glob("maps_*.npy"))
            self.assertEqual(len(files), 1)
            maps = np.load(files[0])
            self.assertEqual(maps.shape, (2048, 1, 32, 32))
            self.assertTrue(np.all(maps == 7))

    def test_smaller_inference_context_preserves_maps_and_noise_rng(self) -> None:
        for map_size in (16, 17, 18, 31, 32, 33, 34, 35, 36):
            with self.subTest(map_size=map_size), TemporaryDirectory() as directory:
                root = Path(directory)
                source_dir = root / "source"
                source_dir.mkdir()
                source_maps = np.random.default_rng(3).normal(0, 0.01, (3, 1, map_size, map_size)).astype(np.float32)
                np.save(source_dir / "maps_000000.npy", source_maps)
                model = ZnUpscalerModel(z_level=1, layer_channels=(4, 4, 4, 4)).eval()
                source = MapBatchSource(source_dir, map_size=map_size)
                torch.manual_seed(7)
                expected = []
                with torch.inference_mode():
                    for _ in range(2):
                        parents = torch.from_numpy(source.take(2))
                        noise = torch.randn(2, 4, map_size, map_size)
                        expected.append(center_crop(model(parents, noise), map_size))
                expected_rng = torch.get_rng_state()
                observed_sizes = []

                def observe_inputs(module, inputs):
                    parent, noise = inputs
                    self.assertEqual(parent.shape[-2:], noise.shape[-2:])
                    observed_sizes.append(parent.shape[-1])

                hook = model.register_forward_pre_hook(observe_inputs)
                torch.manual_seed(7)
                try:
                    with patch("terrain_higan.projects.zn_map_upscaler.generate_input_maps.load_zn_map_upscaler", return_value=model):
                        generate_maps(
                            z_level=1, level_name="z1", total_samples=4,
                            batch_size=2, file_batch_size=4, map_size=map_size,
                            mode="rebuild", output_dir=root / "output", source_dir=source_dir,
                            model_dir=root / "models", z0_model_path=root / "unused.pt2",
                            device=torch.device("cpu"), progress=False,
                        )
                finally:
                    hook.remove()
                actual = torch.from_numpy(np.load(root / "output/maps_000000.npy"))
                torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertEqual(len(observed_sizes), 2)
                for size in observed_sizes:
                    self.assertEqual((map_size - size) % 2, 0)
                    self.assertLessEqual(size, map_size)
                    self.assertIn(2 * size - 16, range(map_size, map_size + 4))

    def test_training_resources_are_collected_before_materialization(self) -> None:
        class TrainingResource:
            pass

        references = []

        def train_and_save(config):
            resource = TrainingResource()
            resource.cycle = resource
            references.append(weakref.ref(resource))

        def materialize(config, device):
            self.assertIsNone(references[0]())
            self.assertEqual(device, torch.device("cpu"))

        with patch.object(train, "_train_and_save", side_effect=train_and_save), \
                patch.object(train, "materialize_trained_maps", side_effect=materialize) as materialization, \
                patch.object(torch.cuda, "empty_cache") as empty_cache:
            train.train_zn_map_upscaler(ZnUpscalerConfig(device="cpu"))
        materialization.assert_called_once()
        empty_cache.assert_not_called()


def test_map_batch_source_reads_across_batch_boundaries(tmp_path) -> None:
    np.save(tmp_path / "maps_000000.npy", np.full((2, 1, 2, 2), 1.0, dtype=np.float32))
    np.save(tmp_path / "maps_000001.npy", np.full((2, 1, 2, 2), 2.0, dtype=np.float32))

    source = MapBatchSource(tmp_path, map_size=2, start_sample=1)

    maps = source.take(4)

    assert maps.shape == (4, 1, 2, 2)
    assert maps[:, 0, 0, 0].tolist() == [1.0, 2.0, 2.0, 1.0]


def test_center_crop_keeps_a_fixed_map_size() -> None:
    maps = torch.arange(6 * 8).reshape(1, 1, 6, 8)

    cropped = center_crop(maps, 4)

    assert torch.equal(cropped, maps[..., 1:5, 2:6])


def test_rebuild_removes_existing_batches(tmp_path) -> None:
    np.save(tmp_path / "maps_000000.npy", np.zeros((1, 1, 2, 2), dtype=np.float32))

    existing_samples = _prepare_output_dir(tmp_path, "rebuild", 2)

    assert existing_samples == 0
    assert not tmp_path.exists()


def test_sample_patch_means_returns_requested_samples_per_map(tmp_path) -> None:
    maps = np.stack(
        (
            np.arange(25, dtype=np.float32).reshape(1, 5, 5),
            np.arange(25, 50, dtype=np.float32).reshape(1, 5, 5),
        )
    )
    map_path = tmp_path / "maps_000000.npy"
    np.save(map_path, maps)

    patch_means, map_count = _sample_patch_means(
        [map_path],
        patch_size=3,
        patches_per_map=4,
        random_generator=np.random.default_rng(1),
    )

    assert map_count == 2
    assert patch_means.shape == (8,)
    assert np.all((patch_means >= 6.0) & (patch_means <= 43.0))

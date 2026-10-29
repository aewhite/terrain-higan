import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import onnx
import pytest
import torch
from safetensors.torch import load_file

from terrain_higan.etopo.normalization import ELEVATION_MEAN_M, ELEVATION_STD_M
from terrain_higan.projects.z0_generator.config import Z0GeneratorConfig
from terrain_higan.projects.z0_generator.export import export_z0_generator
from terrain_higan.projects.z0_generator.models import Z0Generator
from terrain_higan.projects.z0_generator.train import _probe_generator_export


pytestmark = pytest.mark.export


class Z0GeneratorExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_dir = TemporaryDirectory()
        cls.root = Path(cls.temporary_dir.name)
        cls.config = Z0GeneratorConfig(
            generated_size=48,
            model_dir=cls.root / "models",
            run_dir=cls.root / "runs" / "z0_test",
        )
        cls.generator = Z0Generator(cls.config.generator_output_clip_range)
        cls.initial_state = {
            name: values.clone() for name, values in cls.generator.state_dict().items()
        }
        for parameter in cls.generator.parameters():
            parameter.grad = torch.ones_like(parameter)
        try:
            cls.paths = export_z0_generator(cls.generator, cls.config)
        except Exception:
            cls.temporary_dir.cleanup()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_dir.cleanup()

    def test_artifact_names_metadata_and_relative_latest(self) -> None:
        assert self.paths == (
            self.config.state_dict_path,
            self.config.pt2_path,
            self.config.onnx_path,
            self.config.metadata_path,
        )
        assert [path.name for path in self.paths] == [
            "generator.safetensors", "generator.pt2", "generator.onnx", "metadata.json",
        ]
        assert all(path.stat().st_size > 0 for path in self.paths)
        latest = self.config.model_dir / "latest"
        assert latest.is_symlink()
        assert latest.readlink() == Path("z0_test")
        assert latest.resolve() == self.config.model_run_dir
        assert not list(self.config.model_run_dir.glob(".*.tmp"))

        model = onnx.load(self.config.onnx_path)
        onnx.checker.check_model(model)
        assert [value.name for value in model.graph.input] == ["noise"]
        assert [value.name for value in model.graph.output] == ["elevation_normalized"]
        dimensions = model.graph.input[0].type.tensor_type.shape.dim
        assert dimensions[1].dim_value == 1
        assert all(dimensions[index].dim_param for index in (0, 2, 3))
        assert len({dimensions[index].dim_param for index in (0, 2, 3)}) == 3
        metadata = json.loads(self.config.metadata_path.read_text())
        assert metadata == {
            "format_version": 2,
            "model_type": "z0_generator",
            "artifacts": {
                "state_dict": "generator.safetensors", "pt2": "generator.pt2", "onnx": "generator.onnx",
            },
            "input": {
                "name": "noise", "dtype": "float32", "shape": ["batch", 1, "height", "width"],
                "distribution": "standard_normal", "minimum_height": 48, "minimum_width": 48,
            },
            "output": {
                "name": "elevation_normalized", "dtype": "float32",
                "shape": ["batch", 1, "height - generator_shrink", "width - generator_shrink"],
                "units": "normalized",
            },
            "elevation_normalization": {"mean_m": ELEVATION_MEAN_M, "std_m": ELEVATION_STD_M},
            "generator_shrink": 16,
            "torch_version": torch.__version__,
            "onnx_opset_versions": {
                opset.domain or "ai.onnx": opset.version for opset in model.opset_import
            },
            "export_example_noise_shape": [2, 1, 64, 64],
        }

    def test_export_preserves_live_model_and_checkpoint_values(self) -> None:
        saved_state = load_file(self.config.state_dict_path)
        assert saved_state.keys() == self.initial_state.keys()
        assert self.generator.training
        for name, value in self.generator.state_dict().items():
            torch.testing.assert_close(value, self.initial_state[name], rtol=0, atol=0)
            torch.testing.assert_close(saved_state[name], value, rtol=0, atol=0)
        for parameter in self.generator.parameters():
            assert parameter.device.type == "cpu"
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)

    def test_pt2_supports_minimum_and_rectangular_dynamic_shapes(self) -> None:
        program = torch.export.load(self.config.pt2_path)
        assert all(name.startswith("generator._network.") for name in program.state_dict)
        exported_model = program.module()
        with torch.inference_mode():
            for shape in ((1, 1, 48, 48), (2, 1, 64, 67), (3, 1, 80, 81)):
                with self.subTest(shape=shape):
                    noise = torch.randn(shape)
                    actual = exported_model(noise)
                    assert actual.shape == (shape[0], 1, shape[2] - 16, shape[3] - 16)
                    torch.testing.assert_close(actual, self.generator(noise))

    def test_failure_cleans_temporary_files_and_keeps_published_model(self) -> None:
        config = Z0GeneratorConfig(model_dir=self.root / "failure", run_dir=Path("z0_failure"))
        previous_dir = config.model_dir / "previous"
        previous_dir.mkdir(parents=True)
        latest = config.model_dir / "latest"
        latest.symlink_to("previous", target_is_directory=True)
        config.model_run_dir.mkdir()
        paths = (config.state_dict_path, config.pt2_path, config.onnx_path, config.metadata_path)
        for path in paths:
            path.write_bytes(b"existing artifact")
        onnx_program = Mock()
        onnx_program.save.side_effect = lambda temporary_path: temporary_path.write_bytes(b"partial onnx")
        with (
            patch("terrain_higan.projects.z0_generator.export._build_exported_programs", return_value=(Mock(), onnx_program)),
            patch(
                "terrain_higan.projects.z0_generator.export.torch.export.save",
                side_effect=lambda program, temporary_path: temporary_path.write_bytes(b"partial pt2"),
            ),
            patch("terrain_higan.projects.z0_generator.export._write_metadata", side_effect=RuntimeError("metadata failed")),
            self.assertRaisesRegex(RuntimeError, "metadata failed"),
        ):
            export_z0_generator(self.generator, config)
        assert latest.readlink() == Path("previous")
        assert not list(config.model_run_dir.glob(".*.tmp"))
        assert all(path.read_bytes() == b"existing artifact" for path in paths)

    def test_probe_checks_artifacts_without_publishing_latest(self) -> None:
        for empty_artifact in (False, True):
            with self.subTest(empty_artifact=empty_artifact):
                written_paths = []

                def write_probe(generator, config, *, output_dir, update_latest):
                    assert generator is self.generator
                    assert config is self.config
                    assert update_latest is False
                    for name in ("generator.safetensors", "generator.pt2", "generator.onnx", "metadata.json"):
                        path = output_dir / name
                        path.write_bytes(b"" if empty_artifact else b"probe")
                        written_paths.append(path)
                    return tuple(written_paths)

                with patch("terrain_higan.projects.z0_generator.train.export_z0_generator", side_effect=write_probe):
                    if empty_artifact:
                        with self.assertRaisesRegex(RuntimeError, "export probe did not write"):
                            _probe_generator_export(self.generator, self.config)
                    else:
                        _probe_generator_export(self.generator, self.config)
                assert len(written_paths) == 4
                assert not written_paths[0].parent.exists()

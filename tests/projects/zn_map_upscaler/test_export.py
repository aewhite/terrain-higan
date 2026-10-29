import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from safetensors.torch import load_file

from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.save import load_zn_map_upscaler, save_zn_map_upscaler


pytestmark = pytest.mark.export


class ZnUpscalerExportTest(unittest.TestCase):
    def test_export_preserves_artifacts_interfaces_and_live_model(self) -> None:
        for z_level in (1, 11):
            with self.subTest(z_level=z_level), TemporaryDirectory() as directory:
                model = ZnUpscalerModel(z_level=z_level, layer_channels=(8, 8, 4, 2))
                for parameter in model.parameters():
                    parameter.grad = torch.full_like(parameter, 0.01)
                original = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
                gradients = [parameter.grad.clone() for parameter in model.parameters()]
                config = ZnUpscalerConfig(z_level=z_level, model_dir=Path(directory))
                paths = save_zn_map_upscaler(model, config)
                self.assertEqual(paths, (config.model_path, config.pt2_path, config.onnx_path))
                self.assertEqual([path.name for path in paths], ["upscaler.safetensors", "upscaler.pt2", "upscaler.onnx"])
                self.assertTrue(all(path.stat().st_size for path in paths))
                self.assertEqual((config.model_dir / config.level_name / "latest").readlink(), Path(config.run_dir.name))
                self.assertEqual(set(config.model_run_dir.iterdir()), set(paths))
                self.assertTrue(model.training)
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
                for parameter, gradient in zip(model.parameters(), gradients):
                    self.assertEqual(parameter.device.type, "cpu")
                    torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
                saved = load_file(paths[0])
                self.assertEqual(saved.keys(), original.keys())
                for name, tensor in saved.items():
                    torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)
                loaded = load_zn_map_upscaler(paths[0])
                self.assertEqual(loaded._z_level, z_level)
                self.assertFalse(loaded.training)
                exported = torch.export.load(paths[1]).module()
                graph = onnx.load(paths[2])
                onnx.checker.check_model(graph)
                self.assertEqual([value.name for value in graph.graph.input], ["input_map", "noise"])
                self.assertEqual([value.name for value in graph.graph.output], ["elevation_normalized"])
                for first, second in zip(graph.graph.input[0].type.tensor_type.shape.dim, graph.graph.input[1].type.tensor_type.shape.dim):
                    if first.dim_param:
                        self.assertEqual(first.dim_param, second.dim_param)
                evaluator = ReferenceEvaluator(graph)
                for shape in ((1, 1, 9, 11), (3, 1, 12, 15)):
                    input_map = torch.randn(shape) * 0.01
                    noise = torch.randn(shape[0], 4, shape[2], shape[3])
                    with torch.inference_mode():
                        expected = loaded(input_map, noise)
                        torch.testing.assert_close(exported(input_map, noise), expected)
                        torch.testing.assert_close(model(input_map, noise)[:, 1:2], expected, rtol=0, atol=0)
                    actual = evaluator.run(None, {"input_map": input_map.numpy(), "noise": noise.numpy()})[0]
                    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-6)

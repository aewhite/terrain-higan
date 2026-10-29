from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file

from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel
from terrain_higan.projects.zn_map_upscaler.save import load_zn_map_upscaler


def _model() -> ZnUpscalerModel:
    return ZnUpscalerModel(z_level=1, layer_channels=(8, 8, 4, 2)).eval()


def _assert_matches_saved_model(
    loaded_model: ZnUpscalerModel,
    model: ZnUpscalerModel,
) -> None:
    assert loaded_model.training is False
    assert loaded_model.state_dict().keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded_model.state_dict()[name], tensor)


class UpscalerCheckpointTest(unittest.TestCase):
    def test_current_topology_in_pt_format(self) -> None:
        with TemporaryDirectory() as directory:
            model = _model()
            model_path = Path(directory) / "z1/latest/upscaler.safetensors"
            model_path.parent.mkdir(parents=True)
            torch.save(model.state_dict(), model_path.with_suffix(".pt"))

            loaded_model = load_zn_map_upscaler(model_path)

            _assert_matches_saved_model(loaded_model, model)

    def test_current_safetensors_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            model = _model()
            model_path = Path(directory) / "z1/latest/upscaler.safetensors"
            model_path.parent.mkdir(parents=True)
            save_file(model.state_dict(), model_path)

            loaded_model = load_zn_map_upscaler(model_path)

            _assert_matches_saved_model(loaded_model, model)
            parents = torch.randn(2, 1, 13, 15) * 0.01
            noise = torch.randn(2, 4, 13, 15)
            torch.testing.assert_close(loaded_model(parents, noise), model(parents, noise), rtol=0, atol=0)

    def test_obsolete_topology_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            state_dict = _model().state_dict()
            state_dict["_network.0.4.weight"] = state_dict.pop("_network.0.5.weight")
            state_dict["_network.0.6.weight"] = state_dict.pop("_network.0.7.weight")
            model_path = Path(directory) / "upscaler.safetensors"
            save_file(state_dict, model_path)
            with self.assertRaisesRegex(ValueError, "current upscaler layer spec"):
                load_zn_map_upscaler(model_path, z_level=1)

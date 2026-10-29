import torch
import pytest
from safetensors.torch import save_file

from terrain_higan.inference.models import Upscaler, Z0Generator
from terrain_higan.inference.constants import UPSCALE_SETTINGS, residual_scale
from terrain_higan.inference.weights import load_upscaler
from terrain_higan.etopo.normalization import UPSCALE_STATS
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel


def test_z0_generator_uses_current_checkpoint_layout() -> None:
    model = Z0Generator()
    assert tuple(model(torch.randn(2, 1, 32, 33)).shape) == (2, 1, 16, 17)
    assert "_network.0.0.weight" in model.state_dict()
    assert "_network.1.weight" in model.state_dict()


def test_inference_upscaler_returns_elevation_only() -> None:
    model = Upscaler(checkpoint_zoom=1, layer_channels=(8, 8, 8, 8))
    output = model(torch.randn(2, 1, 16, 17), torch.randn(2, 4, 16, 17), destination_zoom=1)

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (2, 1, 16, 18)


def test_inference_loader_preserves_current_checkpoint(tmp_path) -> None:
    training_model = ZnUpscalerModel(z_level=1, layer_channels=(8, 8, 4, 2)).eval()
    path = tmp_path / "upscaler.safetensors"
    save_file(training_model.state_dict(), path)

    loaded = load_upscaler(path, checkpoint_zoom=1)

    input_map = torch.randn(2, 1, 13, 15) * 0.01
    noise = torch.randn(2, 4, 13, 15)
    torch.testing.assert_close(
        loaded(input_map, noise, destination_zoom=1), training_model(input_map, noise),
        rtol=0, atol=1e-6,
    )


def test_super_inference_settings_match_training_statistics() -> None:
    for zoom in range(11, 21):
        settings = UPSCALE_SETTINGS[zoom]
        stats = UPSCALE_STATS[zoom]
        assert residual_scale(zoom) == stats.residual_divisor
        assert settings.elevation_min == stats.dataset_minimum_value
        assert settings.elevation_max == stats.dataset_maximum_value
        assert settings.residual_min == stats.dataset_minimum_raw_residual
        assert settings.residual_max == stats.dataset_maximum_raw_residual


@pytest.mark.parametrize("zoom", range(1, 21))
def test_inference_upscaler_matches_training_model(zoom: int) -> None:
    training_model = ZnUpscalerModel(z_level=zoom, layer_channels=(8, 8, 8, 8)).eval()
    inference_model = Upscaler(checkpoint_zoom=zoom, layer_channels=(8, 8, 8, 8)).eval()
    inference_model.load_state_dict(training_model.state_dict())
    input_map = torch.randn(1, 1, 16, 17)
    noise = torch.randn(1, 4, 16, 17)

    with torch.inference_mode():
        expected = training_model(input_map, noise)
        actual = inference_model(input_map, noise, destination_zoom=zoom)

    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)

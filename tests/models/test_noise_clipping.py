import torch

from terrain_higan.inference.models import Upscaler, Z0Generator as InferenceZ0Generator
from terrain_higan.projects.z0_generator.models import Z0Generator
from terrain_higan.projects.zn_map_upscaler.models import ZnUpscalerModel


def test_models_clip_noise_inputs_to_the_supported_range() -> None:
    z0_noise = torch.full((1, 1, 17, 17), 6.0)
    expected_z0_noise = z0_noise.clamp(-5.0, 5.0)

    z0_model = Z0Generator(output_clip_range=(-1.0, 1.0))
    assert torch.equal(z0_model(z0_noise), z0_model(expected_z0_noise))

    inference_z0_model = InferenceZ0Generator()
    assert torch.equal(inference_z0_model(z0_noise), inference_z0_model(expected_z0_noise))

    input_map = torch.zeros(1, 1, 16, 16)
    upscaler_noise = torch.full((1, 4, 16, 16), 6.0)
    expected_upscaler_noise = upscaler_noise.clamp(-5.0, 5.0)

    map_upscaler = ZnUpscalerModel(z_level=1, layer_channels=(8, 8, 8, 8))
    assert torch.equal(
        map_upscaler(input_map, upscaler_noise)[:, 1:2],
        map_upscaler(input_map, expected_upscaler_noise)[:, 1:2],
    )

    inference_upscaler = Upscaler(checkpoint_zoom=1, layer_channels=(8, 8, 8, 8))
    assert torch.equal(
        inference_upscaler(input_map, upscaler_noise, destination_zoom=1),
        inference_upscaler(input_map, expected_upscaler_noise, destination_zoom=1),
    )

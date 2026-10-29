import torch
import pytest

from terrain_higan.inference.noise import normal_field, upscaler_noise
from terrain_higan.inference.units import meters_per_pixel


def test_meters_per_pixel_supports_the_full_requested_range() -> None:
    assert meters_per_pixel(10) == 512.0
    assert meters_per_pixel(19) == 1.0
    assert meters_per_pixel(20) == 0.5
    with pytest.raises(ValueError, match="0..20"):
        meters_per_pixel(21)


def test_noise_is_addressed_by_global_coordinates() -> None:
    whole = normal_field(seed=1234, zoom=7, x=-2, y=5, width=8, height=6)
    part = normal_field(seed=1234, zoom=7, x=1, y=7, width=3, height=2)

    assert torch.equal(part, whole[..., 2:4, 3:6])


def test_upscaler_noise_matches_pixel_shuffle_channel_order() -> None:
    noise = upscaler_noise(seed=7, zoom=3, x=10, y=-2, width=2, height=2)
    destination = normal_field(seed=7, zoom=3, x=20, y=-4, width=4, height=4)

    assert torch.equal(noise[:, 0:1], destination[..., 0::2, 0::2])
    assert torch.equal(noise[:, 1:2], destination[..., 0::2, 1::2])
    assert torch.equal(noise[:, 2:3], destination[..., 1::2, 0::2])
    assert torch.equal(noise[:, 3:4], destination[..., 1::2, 1::2])
    assert torch.equal(torch.nn.functional.pixel_shuffle(noise, upscale_factor=2), destination)

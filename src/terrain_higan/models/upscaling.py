"""Terrain upscaling from local slopes with parent-mean preservation."""

import torch
import torch.nn.functional as F


UPSCALER_LAYER_SPECS = "{0}/3 {1}/3/2 shuffle/2 {2}/3 {3}/3"
# Current spec: twice the coarse shrink (2 + 4), plus fine shrink (2 + 2).
UPSCALER_SHRINK = 16


def upscaler_convolution_weights(state_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """Read the four convolution weights defined by the current layer spec."""
    names = ("_network.0.0.weight", "_network.0.2.weight", "_network.0.5.weight", "_network.0.7.weight")
    try:
        weights = tuple(state_dict[name] for name in names)
    except KeyError as error:
        raise ValueError("Checkpoint does not match the current upscaler layer spec") from error
    if any(weight.ndim != 4 for weight in weights):
        raise ValueError("Upscaler convolution weights must be rank-four tensors")
    return weights


def mean_preserving_bilinear_2x(input_map: torch.Tensor) -> torch.Tensor:
    """Upscale N1HW maps with H, W > 1, preserving each parent's 2x2 mean."""
    gx = torch.cat((
        input_map[..., 1:2] - input_map[..., :1],
        (input_map[..., 2:] - input_map[..., :-2]) * 0.5,
        input_map[..., -1:] - input_map[..., -2:-1],
    ), dim=-1)
    gy = torch.cat((
        input_map[..., 1:2, :] - input_map[..., :1, :],
        (input_map[..., 2:, :] - input_map[..., :-2, :]) * 0.5,
        input_map[..., -1:, :] - input_map[..., -2:-1, :],
    ), dim=-2)
    left = input_map - gx * 0.25
    right = input_map + gx * 0.25
    vertical_offset = gy * 0.25
    children = torch.cat((
        left - vertical_offset, right - vertical_offset,
        left + vertical_offset, right + vertical_offset,
    ), dim=1)
    # Pixel shuffle interleaves each parent's four children in row-major order.
    return F.pixel_shuffle(children, upscale_factor=2)

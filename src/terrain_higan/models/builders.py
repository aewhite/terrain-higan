import copy
from collections.abc import Callable, Sequence

import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as apply_spectral_norm

ActivationFactory = Callable[[], nn.Module] | nn.Module


def _new_activation(activation: ActivationFactory) -> nn.Module:
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)
    return activation()


def deep_linear(
    feature_sizes: Sequence[int],
    activation: ActivationFactory = nn.ReLU,
    bias: bool = True,
) -> nn.Sequential:
    if len(feature_sizes) < 2:
        raise ValueError("deep_linear requires at least an input and output size")

    layers: list[nn.Module] = []
    for in_features, out_features in zip(feature_sizes, feature_sizes[1:]):
        layers.append(nn.Linear(in_features, out_features, bias=bias))
        layers.append(_new_activation(activation))

    return nn.Sequential(*layers)


def deep_conv2d(
    input_channels: int,
    layer_specs: str,
    activation: ActivationFactory = nn.ReLU,
    *,
    spectral_norm: bool = False,
    **conv2d_kwargs: object,
) -> nn.Sequential:
    """Build channels/kernel[/dilation] convolutions and shuffle/factor operations."""
    tokens = layer_specs.split()
    if not tokens:
        raise ValueError("deep_conv2d requires at least one layer spec")

    layers: list[nn.Module] = []
    in_channels = input_channels
    for token in tokens:
        parts = token.split("/")
        if parts[0] == "shuffle":
            if len(parts) != 2:
                raise ValueError(f"Invalid shuffle spec {token!r}; expected 'shuffle/factor'")
            factor = int(parts[1])
            if in_channels % (factor * factor):
                raise ValueError(f"shuffle/{factor} requires channels divisible by {factor * factor}")
            layers.append(nn.PixelShuffle(factor))
            in_channels //= factor * factor
            continue
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid Conv2d layer spec {token!r}; expected 'channels/kernel_size[/dilation]'")
        out_channels, kernel_size = (int(value) for value in parts[:2])
        layer_kwargs = conv2d_kwargs
        if len(parts) == 3:
            layer_kwargs = {**conv2d_kwargs, "dilation": int(parts[2])}
        convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            **layer_kwargs,
        )
        if spectral_norm:
            convolution = apply_spectral_norm(convolution)
        layers.append(convolution)
        layers.append(_new_activation(activation))
        in_channels = out_channels

    return nn.Sequential(*layers)

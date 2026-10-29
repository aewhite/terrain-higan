import torch
import torch.nn as nn

from terrain_higan.models import deep_conv2d, deep_linear
from terrain_higan.models.responsive_clipper import ResponsiveClipper


class Z0Generator(nn.Module):
    def __init__(
            self,
            output_clip_range: tuple[float, float],
    ) -> None:
        super().__init__()
        output_min, output_max = output_clip_range
        self._network = nn.Sequential(
            deep_conv2d(1, "128/5 128/5 128/5 64/3 16/3 4/1"),
            nn.Conv2d(4, 1, kernel_size=1),
            ResponsiveClipper(output_min, output_max),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        noise = torch.clamp(noise, -5.0, 5.0)
        return self._network(noise)


class Z0Discriminator(nn.Module):
    def __init__(self, patch_size: int, output_clip: float) -> None:
        super().__init__()
        patch_value_count = patch_size * patch_size
        hidden_sizes = [64, 64, 16, 8]
        self._trunk = deep_linear(
            [patch_value_count, *hidden_sizes],
            activation=nn.LeakyReLU,
        )
        self._score_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 1),
            ResponsiveClipper(-output_clip, output_clip),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        patch_values = patches.flatten(start_dim=1)
        features = self._trunk(patch_values)
        return self._score_head(features).squeeze(1)

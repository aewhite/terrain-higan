"""Public configuration types for terrain inference."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferenceConfig:
    device: torch.device | str = "cpu"
    tile_size: int = 64
    batch_size: int = 1
    cache_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        if isinstance(self.tile_size, bool) or not isinstance(self.tile_size, int) or self.tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(self.cache_bytes, bool) or not isinstance(self.cache_bytes, int) or self.cache_bytes < 0:
            raise ValueError("cache_bytes must be a non-negative integer")


@dataclass(frozen=True)
class Region:
    zoom: int
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.zoom <= 20:
            raise ValueError(f"zoom must be in 0..20, got {self.zoom}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

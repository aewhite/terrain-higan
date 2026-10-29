"""Addressable infinite-field terrain generation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import torch

from ._geometry import (
    child_slices,
    parent_region,
    tile_coordinates,
    validate_coordinate_range,
    z0_input,
)
from .cache import CacheStats, TensorCache
from .config import InferenceConfig, Region
from .noise import normal_field, upscaler_noise
from .weights import checkpoint_path, default_weights_dir, load_upscaler, load_z0


class TerrainWorld:
    """One seeded field with fixed weights/configuration, used by one caller at a time.

    Cache-backed tensors are read-only by convention. Clone before modifying them;
    create another world to change the seed, weights, or execution configuration.
    """

    def __init__(
        self,
        *,
        seed: int,
        z0_model: torch.nn.Module,
        upscalers: Mapping[int, torch.nn.Module],
        config: InferenceConfig | None = None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        self.seed = seed
        self.config = config or InferenceConfig()
        device = torch.device(self.config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self._z0_model = z0_model.to(self.config.device).eval()
        self._upscalers = {zoom: model.to(self.config.device).eval() for zoom, model in upscalers.items()}
        self._cache = TensorCache(self.config.cache_bytes)

    @classmethod
    def from_default_weights(
        cls,
        *,
        seed: int,
        config: InferenceConfig | None = None,
        weights_dir: Path | None = None,
    ) -> "TerrainWorld":
        root = default_weights_dir() if weights_dir is None else Path(weights_dir)
        return cls.from_weights(
            seed=seed,
            z0_path=root / checkpoint_path(0),
            upscaler_paths={zoom: root / checkpoint_path(zoom) for zoom in range(1, 21)},
            config=config,
        )

    @classmethod
    def from_weights(
        cls,
        *,
        seed: int,
        z0_path: Path,
        upscaler_paths: Mapping[int, Path],
        config: InferenceConfig | None = None,
    ) -> "TerrainWorld":
        invalid_zooms = set(upscaler_paths) - set(range(1, 21))
        if invalid_zooms:
            raise ValueError(f"upscaler paths only support checkpoint zooms 1..20, got {sorted(invalid_zooms)}")
        return cls(
            seed=seed,
            z0_model=load_z0(Path(z0_path)),
            upscalers={zoom: load_upscaler(Path(path), checkpoint_zoom=zoom) for zoom, path in upscaler_paths.items()},
            config=config,
        )

    @property
    def cache_stats(self) -> CacheStats:
        return self._cache.stats

    def clear_cache(self) -> None:
        self._cache.clear()

    def generate(self, *, zoom: int, x: int, y: int, width: int, height: int) -> torch.Tensor:
        region = Region(zoom, x, y, width, height)
        validate_coordinate_range(region)
        self._required_models(region.zoom)
        with torch.inference_mode():
            return self._generate_batch([region])[0].to(device="cpu", dtype=torch.float32)

    def generate_regions(self, regions: list[Region]) -> torch.Tensor:
        if not regions:
            raise ValueError("regions must not be empty")
        shape = (regions[0].width, regions[0].height)
        if any((region.width, region.height) != shape for region in regions):
            raise ValueError("generate_regions requires regions with the same width and height")
        for region in regions:
            validate_coordinate_range(region)
            self._required_models(region.zoom)
        outputs: dict[int, torch.Tensor] = {}
        with torch.inference_mode():
            for indexes in _compatible_batches(regions, self.config.batch_size):
                generated = self._generate_batch([regions[index] for index in indexes])
                # Keep accelerator memory bounded by a batch, not the whole request list.
                completed = [value.to(device="cpu", dtype=torch.float32) for value in generated]
                outputs.update(zip(indexes, completed, strict=True))
        return torch.cat([outputs[index] for index in range(len(regions))], dim=0)

    def get_tile(self, *, zoom: int, tile_x: int, tile_y: int) -> torch.Tensor:
        size = self.config.tile_size
        region = Region(zoom, tile_x * size, tile_y * size, size, size)
        key = (region.zoom, tile_x, tile_y)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        terrain = self.generate(**vars(region))
        return self._cache.put(key, terrain)

    def iter_tiles(self, region: Region) -> Iterator[tuple[int, int, torch.Tensor]]:
        for tile_x, tile_y in tile_coordinates(region, self.config.tile_size):
            yield tile_x, tile_y, self.get_tile(zoom=region.zoom, tile_x=tile_x, tile_y=tile_y)

    def _generate_batch(self, regions: list[Region]) -> list[torch.Tensor]:
        """Internal batches are nonempty and have one zoom and one input shape."""
        zoom = regions[0].zoom
        if zoom == 0:
            return self._generate_z0(regions)

        parent_regions = [parent_region(region) for region in regions]
        batches = list(_compatible_batches(parent_regions, self.config.batch_size))
        parents: dict[int, torch.Tensor] = {}
        # Keep the established execution order: all parents, then all child batches.
        for indexes in batches:
            generated = self._generate_batch([parent_regions[index] for index in indexes])
            parents.update(zip(indexes, generated, strict=True))

        outputs: dict[int, torch.Tensor] = {}
        for indexes in batches:
            generated = self._upscale(
                zoom,
                [parent_regions[index] for index in indexes],
                [parents[index] for index in indexes],
            )
            for index, output in zip(indexes, generated, strict=True):
                rows, columns = child_slices(regions[index], parent_regions[index])
                outputs[index] = output[..., rows, columns]
        # A missing result is an internal error, never a silently shortened output batch.
        return [outputs[index] for index in range(len(regions))]

    def _generate_z0(self, regions: list[Region]) -> list[torch.Tensor]:
        noise = torch.cat(
            [normal_field(seed=self.seed, **vars(z0_input(region))) for region in regions]
        )
        # Keep ancestors on the execution device; only public results cross back to CPU.
        generated = self._z0_model(noise.to(self.config.device)).to(dtype=torch.float32)
        return list(generated.split(1))

    def _upscale(
        self, zoom: int, parent_regions: list[Region], parents: list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        parent_batch = torch.cat(parents)
        noise_batch = torch.cat(
            [
                upscaler_noise(
                    seed=self.seed,
                    zoom=zoom,
                    x=region.x,
                    y=region.y,
                    width=region.width,
                    height=region.height,
                )
                for region in parent_regions
            ]
        )
        generated = self._upscalers[zoom](
            parent_batch.to(self.config.device),
            noise_batch.to(self.config.device),
            destination_zoom=zoom,
        ).to(dtype=torch.float32)
        return generated.split(1)

    def _required_models(self, zoom: int) -> None:
        missing = [level for level in range(1, zoom + 1) if level not in self._upscalers]
        if missing:
            raise ValueError(f"Missing upscaler checkpoints for z levels {missing}")


def _compatible_batches(regions: list[Region], batch_size: int) -> Iterator[list[int]]:
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, region in enumerate(regions):
        groups.setdefault((region.zoom, region.width, region.height), []).append(index)
    for indexes in groups.values():
        for start in range(0, len(indexes), batch_size):
            yield indexes[start:start + batch_size]

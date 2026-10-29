import torch
import pytest

from terrain_higan.inference import InferenceConfig, Region, TerrainWorld


class _Z0(torch.nn.Module):
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return noise[..., 8:-8, 8:-8] + self.training * 0.0


class _Upscaler(torch.nn.Module):
    def forward(self, input_map: torch.Tensor, noise: torch.Tensor, *, destination_zoom: int) -> torch.Tensor:
        output = torch.nn.functional.interpolate(input_map, scale_factor=2, mode="nearest")[..., 8:-8, 8:-8]
        return output + noise.mean() * 0.0 + destination_zoom * 0.0 + next(self.parameters(), torch.zeros(())).sum() * 0.0


class _NoiseUpscaler(torch.nn.Module):
    def forward(self, input_map: torch.Tensor, noise: torch.Tensor, *, destination_zoom: int) -> torch.Tensor:
        del input_map, destination_zoom
        return torch.nn.functional.pixel_shuffle(noise, upscale_factor=2)[..., 8:-8, 8:-8]


def _world() -> TerrainWorld:
    return TerrainWorld(seed=19, z0_model=_Z0(), upscalers={level: _Upscaler() for level in range(1, 21)},
                        config=InferenceConfig(tile_size=5, cache_bytes=1024 * 1024))


def test_world_generates_every_supported_level() -> None:
    world = _world()
    for zoom in range(21):
        output = world.generate(zoom=zoom, x=-3, y=2, width=3, height=4)
        assert tuple(output.shape) == (1, 1, 4, 3)


def test_super_level_checkpoint_selection_and_limit() -> None:
    world = _world()
    calls = []
    handles = [
        model.register_forward_hook(lambda _model, _inputs, _output, level=level: calls.append(level))
        for level, model in world._upscalers.items()
    ]
    world.generate(zoom=20, x=-3, y=1, width=2, height=2)
    assert calls == list(range(1, 21))
    for handle in handles:
        handle.remove()
    with pytest.raises(ValueError, match="0..20"):
        world.generate(zoom=21, x=0, y=0, width=1, height=1)
    world._upscalers.pop(11)
    with pytest.raises(ValueError, match="11"):
        world.generate(zoom=12, x=0, y=0, width=1, height=1)
    world._upscalers[11] = _Upscaler()
    world._upscalers.pop(20)
    with pytest.raises(ValueError, match="20"):
        world.generate(zoom=20, x=0, y=0, width=1, height=1)


def test_world_regions_agree_on_overlap_and_tiles_are_cached() -> None:
    world = _world()
    whole = world.generate(zoom=4, x=-4, y=-3, width=9, height=7)
    part = world.generate(zoom=4, x=-1, y=-1, width=3, height=2)
    assert torch.allclose(part, whole[..., 2:4, 3:6])

    first = world.get_tile(zoom=2, tile_x=-1, tile_y=0)
    second = world.get_tile(zoom=2, tile_x=-1, tile_y=0)
    assert first.data_ptr() == second.data_ptr()
    assert world.cache_stats.hits >= 1


def test_tiles_preserve_destination_noise_coordinates() -> None:
    world = TerrainWorld(
        seed=19,
        z0_model=_Z0(),
        upscalers={level: _NoiseUpscaler() for level in range(1, 11)},
        config=InferenceConfig(tile_size=5),
    )

    whole = world.generate(zoom=1, x=0, y=0, width=10, height=5)
    tiled = torch.cat([world.get_tile(zoom=1, tile_x=0, tile_y=0), world.get_tile(zoom=1, tile_x=1, tile_y=0)], dim=-1)

    assert torch.equal(tiled, whole)


def test_generate_regions_preserves_request_order() -> None:
    world = _world()
    regions = [Region(3, 0, 0, 2, 2), Region(3, -2, 1, 2, 2)]
    result = world.generate_regions(regions)

    assert torch.allclose(result[0:1], world.generate(zoom=3, x=0, y=0, width=2, height=2))
    assert torch.allclose(result[1:2], world.generate(zoom=3, x=-2, y=1, width=2, height=2))

"""Weight-free checks for the contracts shared by both terrain runtimes."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from terrain_higan.inference import InferenceConfig, Region, TerrainWorld
from terrain_higan.inference._geometry import child_slices, parent_region, tile_coordinates
from terrain_higan.inference.cache import TensorCache
from terrain_higan.inference.noise import _philox4x32
from terrain_higan.inference.weights import _layer_channels, checkpoint_path

FIXTURES = json.loads((Path(__file__).parents[1] / "fixtures/terrain-contract.json").read_text())


def test_shared_geometry_and_philox() -> None:
    for case in FIXTURES["geometry"]:
        region = Region(**case["region"])
        parent = parent_region(region)
        assert vars(parent) == case["parent"]
        rows, columns = child_slices(region, parent)
        assert [columns.start, rows.start] == case["crop"]
    vector = FIXTURES["philox"]
    result = _philox4x32(
        tuple(np.array(value, dtype=np.uint32) for value in vector["counter"]),
        tuple(np.uint32(value) for value in vector["key"]),
    )
    assert [int(value) for value in result] == vector["output"]


@pytest.mark.parametrize("x", [-3, -2, 0, 1])
@pytest.mark.parametrize("y", [-3, -2, 0, 1])
@pytest.mark.parametrize("width,height", [(1, 1), (2, 3), (3, 2), (4, 4)])
def test_parent_covers_child(x, y, width, height) -> None:
    region = Region(20, x, y, width, height)
    parent = parent_region(region)
    rows, columns = child_slices(region, parent)
    assert 0 <= columns.start <= 1
    assert 0 <= rows.start <= 1
    assert columns.stop <= parent.width * 2 - 16
    assert rows.stop <= parent.height * 2 - 16
    assert 2 * parent.x + 8 + columns.start == x
    assert 2 * parent.y + 8 + rows.start == y


def test_cache_storage_and_accounting() -> None:
    cache = TensorCache(32)
    backing = torch.zeros(8, 1, 2, 2)
    cached = cache.put("a", backing[0:1])
    assert cached.untyped_storage().nbytes() == 16
    assert cache.stats.hits == cache.stats.misses == 0
    assert cache.get("a") is cached
    cache.put("b", torch.ones(1, 1, 2, 2))
    cache.put("c", torch.ones(1, 1, 2, 2))
    assert cache.get("a") is None
    assert cache.stats.bytes == 32
    assert cache.stats.evictions == 1
    cache.put("large", torch.zeros(100))
    assert cache.stats.bytes == 32
    cache.clear()
    assert cache.stats.bytes == 0
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    disabled = TensorCache(0)
    disabled.put("a", backing)
    assert disabled.get("a") is None


class NoiseZ0(torch.nn.Module):
    def forward(self, noise):
        return noise[..., 8:-8, 8:-8]


class NoiseUpscaler(torch.nn.Module):
    def forward(self, parent, noise, *, destination_zoom):
        return torch.nn.functional.pixel_shuffle(noise, 2)[..., 8:-8, 8:-8]


def test_batch_order_shapes_and_cache_metrics() -> None:
    world = TerrainWorld(
        seed=19,
        z0_model=NoiseZ0(),
        upscalers={zoom: NoiseUpscaler() for zoom in range(1, 21)},
        config=InferenceConfig(batch_size=3, tile_size=5),
    )
    origins = [(20, -3, 2), (0, 2, -1), (20, 0, 1), (20, 1, 1), (20, -2, 0)]
    regions = [Region(zoom, x, y, 3, 2) for zoom, x, y in origins]
    batched = world.generate_regions(regions)
    expected = torch.cat([world.generate(**vars(region)) for region in regions])
    assert torch.equal(batched, expected)
    first = world.get_tile(zoom=1, tile_x=-1, tile_y=0)
    assert world.cache_stats.misses == 1
    assert world.cache_stats.hits == 0
    assert world.get_tile(zoom=1, tile_x=-1, tile_y=0) is first
    assert world.cache_stats.hits == 1
    assert list(tile_coordinates(Region(1, -3, -2, 7, 5), 4)) == [
        (-1, -1), (0, -1), (-1, 0), (0, 0),
    ]


def test_weight_paths_and_invalid_convolution_rank() -> None:
    assert checkpoint_path(0).as_posix() == "z0_generator/latest/generator.safetensors"
    assert checkpoint_path(20).as_posix() == "zn_map_upscaler/z20s/latest/upscaler.safetensors"
    state = {f"_network.0.{index}.weight": torch.tensor(1.0) for index in (0, 2, 5, 7)}
    with pytest.raises(ValueError, match="rank-four"):
        _layer_channels(state)


def test_release_and_runtime_use_the_same_model_catalog() -> None:
    script = Path(__file__).parents[2] / "scripts/build_inference_distribution.py"
    spec = importlib.util.spec_from_file_location("release_catalog", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.WEIGHT_PATHS == tuple(checkpoint_path(zoom).as_posix() for zoom in range(21))

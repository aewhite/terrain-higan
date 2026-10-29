import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.image as mpl_image

from terrain_higan.demos.map_to_image import colormaps
from terrain_higan.demos.map_to_image import generate
from terrain_higan.demos.map_to_image.generate import default_seed, encode_seed, generate_normalized_map, resolve_device
from terrain_higan.inference import InferenceConfig, Region


class _World:
    def __init__(self) -> None:
        self.config = InferenceConfig(tile_size=4)

    def iter_tiles(self, region: Region):
        left = region.x // self.config.tile_size
        top = region.y // self.config.tile_size
        right = (region.x + region.width - 1) // self.config.tile_size
        bottom = (region.y + region.height - 1) // self.config.tile_size
        for tile_y in range(top, bottom + 1):
            for tile_x in range(left, right + 1):
                x = tile_x * self.config.tile_size
                y = tile_y * self.config.tile_size
                values = torch.arange(self.config.tile_size ** 2, dtype=torch.float32).reshape(1, 1, 4, 4)
                yield tile_x, tile_y, values + x + y * 100


def test_stitched_map_handles_negative_non_tile_aligned_region(tmp_path) -> None:
    region = Region(4, -3, -2, 7, 5)
    terrain = generate_normalized_map(_World(), region, tmp_path / "map.npy")

    expected = np.empty((5, 7), dtype=np.float32)
    for row, y in enumerate(range(-2, 3)):
        for column, x in enumerate(range(-3, 4)):
            tile_x, tile_y = x // 4, y // 4
            expected[row, column] = (x - tile_x * 4) + 4 * (y - tile_y * 4) + tile_x * 4 + tile_y * 400
    assert np.array_equal(terrain, expected)


def test_elevation_relief_separates_green_land_and_depth_colored_ocean() -> None:
    values = np.array([[-10_000.0, -1.0, 0.0, 500.0, 9_000.0]])
    rgb = colormaps.elevation(values)
    assert rgb[0, 0, 2] > rgb[0, 0, 1] > rgb[0, 0, 0]
    assert rgb[0, 1, 2] > rgb[0, 0, 2]
    assert rgb[0, 2, 1] > rgb[0, 2, 0]
    assert rgb[0, 2, 1] > rgb[0, 2, 2]
    assert rgb[0, 4].min() > 120


def test_hillshade_is_flat_safe_and_northwest_lit() -> None:
    flat = colormaps.compute_hillshade(np.zeros((8, 8), dtype=np.float32))
    east_rising = colormaps.compute_hillshade(np.tile(np.arange(8, dtype=np.float32), (8, 1)))
    west_rising = colormaps.compute_hillshade(np.tile(np.arange(8, dtype=np.float32)[::-1], (8, 1)))
    assert np.allclose(flat, flat[0, 0])
    assert east_rising[4, 4] > west_rising[4, 4]


def test_seed_encoding_and_device_override() -> None:
    assert encode_seed(0) == "0"
    assert encode_seed(61) == "z"
    assert encode_seed(62) == "10"
    assert resolve_device("cpu").type == "cpu"


def test_installed_image_output_remains_relative_to_caller(monkeypatch) -> None:
    monkeypatch.setattr(generate, "find_spec", lambda name: None)
    assert generate.default_output_dir() == Path("runs/map_to_image")


def test_default_device_and_seed(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(None).type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(None).type == "cuda"
    monkeypatch.setattr(generate.time, "time_ns", lambda: 123_456_789_000)
    assert default_seed() == 123_456


def test_realistic_palette_distinguishes_shallow_ocean_and_mountain_bands() -> None:
    rgb = colormaps.realistic(np.array([[-100.0, -25.0, 0.0, 2000.0, 3000.0, 4850.0, 5000.0]]))
    assert rgb[0, 0, 2] < rgb[0, 1, 2]
    assert tuple(rgb[0, 1]) == (142, 213, 229)
    assert rgb[0, 3, 0] == rgb[0, 3, 1] > rgb[0, 3, 2]
    assert rgb[0, 4, 0] > rgb[0, 4, 1] > rgb[0, 4, 2]
    assert rgb[0, 5].min() > rgb[0, 3].min()
    assert rgb[0, 6].min() > rgb[0, 3].min()
    flat_land = colormaps.realistic(np.full((8, 8), 2000.0, dtype=np.float32))
    assert tuple(flat_land[4, 4]) == (133, 133, 128)


def test_cli_writes_exact_arrays_pngs_and_metadata(tmp_path, monkeypatch) -> None:
    from unittest.mock import Mock

    load = Mock(return_value=_World())
    monkeypatch.setattr(generate.TerrainWorld, "from_default_weights", load)
    generate.main([
        "--width", "7", "--height", "5", "--x", "-3", "--y", "-2",
        "--seed", "62", "--z-level", "20", "--device", "cpu",
        "--tile-size", "4", "--output-dir", str(tmp_path), "--weights-dir", str(tmp_path / "custom-weights"),
    ])
    assert load.call_args.kwargs["weights_dir"] == tmp_path / "custom-weights"
    metadata = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert metadata["shape"] == [5, 7]
    assert metadata["encoded_seed"] == "10"
    assert metadata["meters_per_pixel"] == 0.5
    assert np.load(tmp_path / metadata["data_file"]).shape == (5, 7)
    for filename in metadata["images"].values():
        assert mpl_image.imread(tmp_path / filename).shape[:2] == (5, 7)


def test_shared_realistic_palette_stops() -> None:
    fixtures = json.loads((Path(__file__).parents[1] / "fixtures/terrain-contract.json").read_text())
    for case in fixtures["realistic"]:
        # Flat maps isolate the palette from each renderer's border shading.
        actual = colormaps.realistic(np.full((3, 3), case["meters"], dtype=np.float32))
        assert actual[1, 1].tolist() == case["rgb"]


def test_renderers_mark_nonfinite_external_values_black() -> None:
    for name in generate.DEFAULT_COLORMAPS:
        pixels = colormaps.render(name, np.array([[np.nan, np.inf, -np.inf, 0.0]]))
        assert np.array_equal(pixels[0, :3], np.zeros((3, 3), dtype=np.uint8))

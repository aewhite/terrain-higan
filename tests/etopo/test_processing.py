import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

try:
    from terrain_higan.etopo import process_etopo as process
except ModuleNotFoundError as error:
    if error.name not in {"rasterio", "pyproj"}:
        raise
    raise unittest.SkipTest("preprocessing checks require the geospatial dependencies") from error


class ProcessingTests(unittest.TestCase):
    def test_extract_patch_passes_warp_options_to_gdal(self):
        def open_warped(source, **options):
            warped = process.rasterio.vrt.WarpedVRT(source, **options)
            self.assertEqual(warped.warp_extras["SAMPLE_GRID"], "YES")
            self.assertEqual(warped.warp_extras["SOURCE_EXTRA"], "100")
            self.assertNotIn("warp_extras", warped.warp_extras)
            return warped

        with process.rasterio.io.MemoryFile() as memory:
            with memory.open(
                driver="GTiff", width=72, height=36, count=1, dtype="float32",
                crs="EPSG:4326", transform=process.rasterio.transform.from_origin(-180, 90, 5, 5),
            ) as target:
                target.write(process.np.full((1, 36, 72), 1234, dtype=process.np.float32))
            with (
                memory.open() as source,
                patch.object(process, "_worker_vrt", source),
                patch.object(process, "random_lat_lon", return_value=(0, 0)),
                patch.object(process, "WarpedVRT", side_effect=open_warped),
            ):
                result = process.extract_patch(output_size=32)

        torch.testing.assert_close(result, torch.full((32, 32), 1234.0))

    def test_selected_levels_reach_extraction_shards_and_previews(self):
        batch = torch.ones(1, 1, 3072, 3072)
        for min_z, max_z in [(0, 3), (2, 3), (3, 3), (10, 10)]:
            with self.subTest(min_z=min_z, max_z=max_z), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                writer = Mock()
                with (
                    patch.object(process, "DataLoader", return_value=[batch]),
                    patch.object(process, "init_worker"),
                    patch.object(process.torch.cuda, "is_available", return_value=False),
                    patch.object(process, "SummaryWriter", return_value=writer),
                    patch.object(process, "build_sample", wraps=process.build_sample) as build_sample,
                ):
                    process.main([
                        "--base-dir", str(root), "--device", "cpu", "--workers", "0", "--steps", "1",
                        "--min-z", str(min_z), "--max-z", str(max_z),
                    ])

                levels = list(range(min_z, max_z + 1))
                self.assertEqual([call.kwargs["zoom_level"] for call in build_sample.call_args_list], levels)
                self.assertEqual({entry.name for entry in root.iterdir()}, {f"z{level}" for level in levels})
                self.assertEqual(writer.add_image.call_count, len(levels) + 1)
                for level, call in zip(levels, writer.add_image.call_args_list[1:]):
                    self.assertEqual(call.args[0], f"sample/z{level}")
                    patch_size = call.args[1].shape[-1]
                    shard = root / f"z{level}" / f"{patch_size}x{patch_size}" / "shard-000000"
                    self.assertGreater(shard.stat().st_size, 0)
                    self.assertEqual(shard.stat().st_size % (patch_size * patch_size * 4), 0)

    def test_invalid_zoom_ranges_are_rejected(self):
        for min_z, max_z in [(-1, 3), (0, 11), (4, 3)]:
            with self.subTest(min_z=min_z, max_z=max_z), self.assertRaises(SystemExit) as error:
                process.parse_arguments(["--min-z", str(min_z), "--max-z", str(max_z)])
            self.assertEqual(error.exception.code, 2)

"""Release staging checks that need no model runtime or real checkpoints."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_inference_distribution.py"
SPEC = importlib.util.spec_from_file_location("inference_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class DistributionTests(unittest.TestCase):
    def test_weight_cli_defaults_ignore_environment(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {"TERRAIN_HIGAN_WEIGHTS_DIR": "alternate-weights"}):
            self.assertEqual(release.parse_arguments([]).weights_dir, release.ROOT / ".local/weights")
            self.assertEqual(release.parse_arguments(["--weights-dir", "explicit"]).weights_dir, Path("explicit"))

    def test_missing_weights_fail_before_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                release.stage_project(root, root / "absent")
            self.assertEqual(list(root.iterdir()), [])

    def test_staging_is_self_contained_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "weights"
            for name in release.WEIGHT_PATHS:
                path = weights / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                path.with_suffix(".onnx").write_bytes(b"not shipped")
            stage = root / "stage"
            stage.mkdir()
            release.stage_project(stage, weights)
            package = stage / "src"
            manifest_path = package / "terrain_higan/inference/weights/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["weights"]), 21)
            self.assertFalse((package / "terrain_higan/projects").exists())
            self.assertTrue((package / "terrain_higan/models/upscaling.py").is_file())
            self.assertFalse(list(stage.rglob("*.onnx")))
            wheel = root / "test.whl"

            def make_wheel():
                with zipfile.ZipFile(wheel, "w") as archive:
                    for member in package.rglob("*"):
                        if member.is_file():
                            archive.write(member, member.relative_to(package).as_posix())

            make_wheel()
            release.inspect_wheel(wheel)
            (package / "terrain_higan/inference/weights" / release.WEIGHT_PATHS[0]).write_bytes(b"corrupt")
            make_wheel()
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                release.inspect_wheel(wheel)


if __name__ == "__main__":
    unittest.main()

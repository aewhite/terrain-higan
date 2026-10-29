from pathlib import Path
from unittest.mock import Mock

import pytest

from terrain_higan.inference import weights
from terrain_higan.inference.world import TerrainWorld


def test_explicit_weights_override_stable_source_defaults(tmp_path, monkeypatch):
    root = Path(weights.__file__).resolve().parents[3]
    monkeypatch.setenv("TERRAIN_HIGAN_WEIGHTS_DIR", "alternate-weights")
    monkeypatch.chdir(tmp_path)
    assert weights.default_weights_dir() == root / ".local/weights"
    load = Mock()
    monkeypatch.setattr(TerrainWorld, "from_weights", load)
    TerrainWorld.from_default_weights(seed=1, weights_dir=tmp_path / "explicit")
    assert load.call_args.kwargs["z0_path"] == tmp_path / "explicit" / weights.checkpoint_path(0)


def test_installed_weights_are_self_contained_and_verified(tmp_path, monkeypatch):
    module_path = tmp_path / "lib/site-packages/terrain_higan/inference/weights.py"
    packaged = module_path.with_name("weights")
    packaged.mkdir(parents=True)
    monkeypatch.setattr(weights, "__file__", str(module_path))
    verify = Mock()
    monkeypatch.setattr(weights, "_verify_packaged_weights", verify)
    assert weights.default_weights_dir() == packaged
    verify.assert_called_once_with(packaged)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TERRAIN_HIGAN_WEIGHTS_DIR", "external")
    assert weights.default_weights_dir() == packaged
    assert verify.call_count == 2


def test_installed_package_without_weights_does_not_guess_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "__file__", str(tmp_path / "lib/site-packages/terrain_higan/inference/weights.py"))
    with pytest.raises(FileNotFoundError, match="No packaged weights"):
        weights.default_weights_dir()

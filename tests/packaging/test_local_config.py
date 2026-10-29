from pathlib import Path
import shutil

import pytest


def test_personal_python_configuration_is_excluded_from_distributions(tmp_path, monkeypatch):
    setuptools = pytest.importorskip("setuptools")
    root = Path(__file__).resolve().parents[2]
    package = tmp_path / "src/terrain_higan"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "paths.py").write_text("PROJECT_ROOT = None\n")
    (tmp_path / "src/overrides.py").write_text('DATA_DIR = "/private/training-data"\n')
    for name in ("pyproject.toml", "MANIFEST.in", "README.md"):
        shutil.copy2(root / name, tmp_path / name)
    monkeypatch.chdir(tmp_path)
    distribution = setuptools.Distribution()
    distribution.script_name = "setup.py"
    distribution.parse_config_files()
    build = distribution.get_command_obj("build_py")
    build.ensure_finalized()
    modules = build.find_all_modules()
    assert any(package_name == "terrain_higan" and module_name == "paths" for package_name, module_name, filename in modules)
    assert all(module_name != "overrides" for package_name, module_name, filename in modules)
    metadata = distribution.get_command_obj("egg_info")
    metadata.ensure_finalized()
    metadata.run()
    assert "src/overrides.py" not in metadata.filelist.files

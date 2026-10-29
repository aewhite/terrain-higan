import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    shutil.copytree(
        ROOT / "src/terrain_higan", root / "src/terrain_higan",
        ignore=shutil.ignore_patterns("__pycache__", "assets", "node_modules"),
    )
    return root


PATHS_CODE = """
import json
from terrain_higan.paths import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, DEFAULT_RUNS_DIR, DEFAULT_ETOPO_VRT
from terrain_higan.etopo.dataloader import DEFAULT_ETOPO_BASE_DIR
from terrain_higan.etopo.fixup_zero_samples import DEFAULT_BASE_DIR
from terrain_higan.etopo.shuffle_shards import DEFAULT_BASE_DIR as SHUFFLE_BASE_DIR
from terrain_higan.projects.z0_generator.config import Z0GeneratorConfig
from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig
from terrain_higan.demos.map_to_image.generate import default_output_dir
from terrain_higan.inference.weights import default_weights_dir
config = ZnUpscalerConfig()
print(json.dumps([str(value) for value in (
    DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, DEFAULT_RUNS_DIR, DEFAULT_ETOPO_VRT,
    DEFAULT_ETOPO_BASE_DIR, DEFAULT_BASE_DIR, SHUFFLE_BASE_DIR,
    Z0GeneratorConfig().model_dir, config.model_dir, config.z0_pt2_path,
    config.maps_dir, config.analysis_dir, config.run_dir.parent,
    default_output_dir(), default_weights_dir(),
)]))
"""


@pytest.mark.parametrize("overrides", [
    {},
    {"DATA_DIR": "custom-data"},
    {
        "DATA_DIR": "~/terrain-data",
        "MODELS_DIR": "custom-models",
        "RUNS_DIR": "custom-runs",
        "ETOPO_VRT": "custom-source/elevation.vrt",
    },
])
def test_paths_are_aligned_and_independent_of_working_directory(tmp_path, checkout, overrides):
    environment = {name: value for name, value in os.environ.items() if not name.startswith("TERRAIN_HIGAN_")}
    (checkout / "src/overrides.py").write_text("\n".join(f"{name} = {value!r}" for name, value in overrides.items()))
    environment.update(PYTHONPATH=str(checkout / "src"), TERRAIN_HIGAN_DATA_DIR="ignored-environment-data")
    result = subprocess.run(
        [sys.executable, "-c", PATHS_CODE], cwd=tmp_path,
        env=environment, capture_output=True, text=True, check=True,
    )
    actual = [Path(value) for value in json.loads(result.stdout)]
    data = checkout / Path(overrides.get("DATA_DIR", "data")).expanduser()
    models = checkout / overrides["MODELS_DIR"] if "MODELS_DIR" in overrides else data / "models"
    runs = checkout / overrides.get("RUNS_DIR", "runs")
    vrt = checkout / overrides["ETOPO_VRT"] if "ETOPO_VRT" in overrides else data / "etopo2022_15s/etopo_2022_15s_bedpref.vrt"
    datasets = data / "datasets/etopo2022/15s"
    assert actual == [
        data, models, runs, vrt, datasets, datasets, datasets,
        models / "z0_generator", models / "zn_map_upscaler",
        models / "z0_generator/latest/generator.pt2", data / "maps",
        data / "analysis", runs, runs / "map_to_image",
        checkout / ".local/weights",
    ]


@pytest.mark.parametrize("environment_data", [None, "explicit-data"])
def test_python_overrides_load_automatically_in_synced_source(tmp_path, checkout, environment_data):
    (checkout / "src/overrides.py").write_text(
        'DATA_DIR = "personal-data"\nRUNS_DIR = "~/personal-runs"\n'
        'MODELS_DIR = "personal-models"\nETOPO_VRT = "personal-source/elevation.vrt"\n'
        'TENSORBOARD_HOST = "configured-host"\nTENSORBOARD_PORT = 6012\n'
    )
    environment = {name: value for name, value in os.environ.items() if not name.startswith("TERRAIN_HIGAN_")}
    environment["PYTHONPATH"] = str(checkout / "src")
    if environment_data is not None:
        environment["TERRAIN_HIGAN_DATA_DIR"] = environment_data
    result = subprocess.run(
        [sys.executable, "-c", PATHS_CODE], cwd=tmp_path, env=environment,
        capture_output=True, text=True, check=True,
    )
    actual = [Path(value) for value in json.loads(result.stdout)]
    data = checkout / "personal-data"
    models = checkout / "personal-models"
    runs = Path.home() / "personal-runs"
    assert actual[:4] == [data, models, runs, checkout / "personal-source/elevation.vrt"]
    assert actual[4:7] == [data / "datasets/etopo2022/15s"] * 3
    assert actual[7:10] == [models / "z0_generator", models / "zn_map_upscaler", models / "z0_generator/latest/generator.pt2"]
    assert actual[12:14] == [runs, runs / "map_to_image"]
    launch_code = '''
from unittest.mock import Mock, patch
from terrain_higan.tensorboard_utils import tensorboard_helpers as helpers
with patch.object(helpers, "is_tensorboard_running", return_value=False), patch.object(helpers.time, "sleep"), patch.object(helpers.subprocess, "Popen", return_value=Mock(poll=Mock(return_value=None))) as launch:
    helpers.start_tensorboard_detached("test-runs")
    command = launch.call_args.args[0]
    assert command[command.index("--host") + 1] == "configured-host"
    assert command[command.index("--port") + 1] == "6012"
'''
    subprocess.run([sys.executable, "-c", launch_code], cwd=tmp_path, env=environment, check=True)


def test_broken_python_override_is_reported_instead_of_ignored(tmp_path, checkout):
    (checkout / "src/overrides.py").write_text('raise RuntimeError("broken local configuration")\n')
    result = subprocess.run(
        [sys.executable, "-c", "import terrain_higan.paths"], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(checkout / "src")}, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "broken local configuration" in result.stderr


def test_tensorboard_uses_current_interpreter_and_explicit_overrides(tmp_path, monkeypatch):
    from terrain_higan.tensorboard_utils import tensorboard_helpers as helpers

    monkeypatch.setenv("TERRAIN_HIGAN_TENSORBOARD_HOST", "ignored-host")
    monkeypatch.setenv("TERRAIN_HIGAN_TENSORBOARD_PORT", "9999")
    monkeypatch.setattr(helpers, "is_tensorboard_running", lambda port: False)
    monkeypatch.setattr(helpers.time, "sleep", lambda duration: None)
    launch = Mock(return_value=Mock(poll=Mock(return_value=None)))
    monkeypatch.setattr(helpers.subprocess, "Popen", launch)
    helpers.start_tensorboard_detached(tmp_path / "runs", host="configured-host", port=6010)
    command = launch.call_args.args[0]
    assert command[:3] == [sys.executable, "-m", "tensorboard.main"]
    assert command[command.index("--logdir") + 1] == str(tmp_path / "runs")
    assert command[command.index("--host") + 1] == "configured-host"
    assert command[command.index("--port") + 1] == "6010"
    assert launch.call_args.kwargs["start_new_session"] is True
    helpers.start_tensorboard_detached(tmp_path / "other", host="127.0.0.1", port=6011)
    command = launch.call_args.args[0]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "6011"


def test_preprocessing_cli_passes_vrt_to_workers_and_respects_gdal_override(tmp_path, monkeypatch):
    pytest.importorskip("rasterio", reason="preprocessing checks require the geospatial dependencies")
    pytest.importorskip("pyproj", reason="preprocessing checks require the geospatial dependencies")
    from terrain_higan.etopo import process_etopo as process

    vrt = tmp_path / "source.vrt"
    opened = Mock()
    monkeypatch.setattr(process.rasterio, "open", opened)
    monkeypatch.setattr(process, "_worker_vrt", None)
    monkeypatch.setattr(process, "_worker_call_count", 0)
    monkeypatch.setattr(process, "ELEVATION_DATA", None)
    monkeypatch.setattr(process.random, "seed", Mock())
    monkeypatch.setattr(process.np.random, "seed", Mock())
    monkeypatch.setenv("GDAL_CACHEMAX", "16")
    monkeypatch.setenv("GDAL_BLOCK_CACHE_SIZE", "67108864")
    monkeypatch.setenv("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    loader = Mock(return_value=[process.torch.zeros(1)])
    monkeypatch.setattr(process, "DataLoader", loader)
    monkeypatch.setattr(process, "ETOPOpatchDataset", Mock())
    monkeypatch.setattr(process, "ShardManager", Mock())
    monkeypatch.setattr(process, "SummaryWriter", Mock())
    monkeypatch.setattr(process, "extract_samples", Mock(return_value=[]))
    monkeypatch.setattr(process, "log_images_to_tensorboard", Mock())
    process.main(["--vrt", str(vrt), "--workers", "2", "--steps", "1", "--device", "cpu"])
    loader.call_args.kwargs["worker_init_fn"](0)
    opened.assert_called_once_with(vrt)
    assert process.ELEVATION_DATA == vrt
    assert os.environ["GDAL_CACHEMAX"] == "16"


def test_resource_cli_controls_reach_upscaler_config(monkeypatch):
    from terrain_higan.projects.zn_map_upscaler import train

    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    launch = Mock()
    monkeypatch.setattr(train, "train_zn_map_upscaler", launch)
    train.main([
        "--device", "cuda:1", "--input-maps-per-batch", "1024",
        "--patches-per-batch", "128",
        "--input-map-loading-workers", "1", "--input-map-generation-batch-size", "64",
    ])
    config = launch.call_args.args[0]
    assert config.device == "cuda:1"
    assert config.input_maps_per_batch == 1024
    assert config.patches_per_batch == 128
    assert config.input_map_loading_workers == 1
    assert config.input_map_generation_batch_size == 64


@pytest.mark.parametrize("selected_count", [0, -1, 9])
def test_upscaler_selected_batch_must_fit_candidate_population(selected_count):
    from terrain_higan.projects.zn_map_upscaler.config import ZnUpscalerConfig

    with pytest.raises(ValueError, match="patches_per_batch"):
        ZnUpscalerConfig(
            input_maps_per_batch=8, input_map_total_samples=8,
            patches_per_batch=selected_count,
        )
    config = ZnUpscalerConfig(
        input_maps_per_batch=8, input_map_total_samples=8, patches_per_batch=8,
    )
    assert config.patches_per_batch == config.input_maps_per_batch


def test_upscaler_cli_uses_config_defaults_and_explicit_level_overrides(monkeypatch):
    from terrain_higan.projects.zn_map_upscaler import train

    monkeypatch.setattr(train.torch.cuda, "is_available", lambda: True)
    launch = Mock()
    monkeypatch.setattr(train, "train_zn_map_upscaler", launch)
    defaults = train.ZnUpscalerConfig()
    train.main([])
    config = launch.call_args.args[0]
    assert config.z_level == defaults.z_level
    assert config.minimum_steps == defaults.minimum_steps
    train.main(["--z-level", "4", "--minimum-steps", "12345"])
    config = launch.call_args.args[0]
    assert config.z_level == 4
    assert config.minimum_steps == 12345


def test_training_cli_forwards_tensorboard_options(monkeypatch):
    from terrain_higan import main

    monkeypatch.setattr(main, "setup_pytorch", Mock())
    monkeypatch.setattr(main, "train_zn_map_upscaler", Mock())
    launch = Mock()
    monkeypatch.setattr(main.tensorboard_helpers, "start_tensorboard_detached", launch)
    main.main(["--tensorboard-host", "127.0.0.1", "--tensorboard-port", "6013"])
    launch.assert_called_once_with(host="127.0.0.1", port=6013)


@pytest.mark.parametrize("arguments,overrides,expected_source,expected_destination", [
    ([], {}, None, None),
    (["explicit-models"], {}, "explicit-models", None),
    (["trainer:/path/to/models"], {}, "trainer:/path/to/models", None),
    (["explicit-models", "explicit-weights"], {}, "explicit-models", "explicit-weights"),
    ([], {"TERRAIN_HIGAN_WEIGHTS_SOURCE": "trainer:/path/to/models"}, "trainer:/path/to/models", None),
    ([], {"TERRAIN_HIGAN_WEIGHTS_DIR": "environment-weights"}, None, "environment-weights"),
    (
        [],
        {"TERRAIN_HIGAN_WEIGHTS_SOURCE": "models with spaces", "TERRAIN_HIGAN_WEIGHTS_DIR": "weights with spaces"},
        "models with spaces", "weights with spaces",
    ),
    (
        ["explicit-models"],
        {"TERRAIN_HIGAN_WEIGHTS_SOURCE": "environment-models", "TERRAIN_HIGAN_WEIGHTS_DIR": "environment-weights"},
        "explicit-models", "environment-weights",
    ),
    (
        ["explicit-models", "explicit-weights"],
        {"TERRAIN_HIGAN_WEIGHTS_SOURCE": "environment-models", "TERRAIN_HIGAN_WEIGHTS_DIR": "environment-weights"},
        "explicit-models", "explicit-weights",
    ),
    ([], {"TERRAIN_HIGAN_WEIGHTS_SOURCE": "", "TERRAIN_HIGAN_WEIGHTS_DIR": ""}, None, None),
])
def test_weight_sync_uses_shared_paths_without_contacting_remote_hosts(
    tmp_path, arguments, overrides, expected_source, expected_destination,
):
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "sync_latest_weights.sh"
    shutil.copy2(ROOT / "scripts/sync_latest_weights.sh", script)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name, content in {
        "rsync": '#!/bin/sh\nprintf "%s\\n" "$@" > "$SYNC_ARGUMENTS"\n',
        "node": '#!/bin/sh\nprintf "%s\\n" "$@" > "$SYNC_NODE_ARGUMENTS"\n',
    }.items():
        executable = binaries / name
        executable.write_text(content)
        executable.chmod(0o755)
    environment = {name: value for name, value in os.environ.items() if not name.startswith("TERRAIN_HIGAN_")}
    environment.update(
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        SYNC_ARGUMENTS=str(tmp_path / "arguments"),
        SYNC_NODE_ARGUMENTS=str(tmp_path / "node-arguments"),
    )
    environment.update(overrides)
    subprocess.run(["bash", str(script), *arguments], cwd=tmp_path, env=environment, check=True)
    source = expected_source if expected_source is not None else str(project / "data/models")
    destination = str(tmp_path / expected_destination if expected_destination is not None else project / ".local/weights")
    assert (tmp_path / "arguments").read_text().splitlines()[-2:] == [source + "/", destination + "/"]
    assert (tmp_path / "node-arguments").read_text().splitlines()[-2:] == ["--weights-dir", destination]

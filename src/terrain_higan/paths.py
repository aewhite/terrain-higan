from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CONFIG_PATH = PROJECT_ROOT / "src" / "overrides.py"
LOCAL_CONFIG = runpy.run_path(str(LOCAL_CONFIG_PATH)) if LOCAL_CONFIG_PATH.is_file() else {}


def setting(name: str, default):
    return LOCAL_CONFIG.get(name, default)


def configured_path(name: str, default: Path) -> Path:
    value = Path(setting(name, default)).expanduser()
    return PROJECT_ROOT / value


DEFAULT_DATA_DIR = configured_path("DATA_DIR", PROJECT_ROOT / "data")
DEFAULT_RUNS_DIR = configured_path("RUNS_DIR", PROJECT_ROOT / "runs")
DEFAULT_MODELS_DIR = configured_path("MODELS_DIR", DEFAULT_DATA_DIR / "models")
DEFAULT_ETOPO_VRT = configured_path("ETOPO_VRT", DEFAULT_DATA_DIR / "etopo2022_15s" / "etopo_2022_15s_bedpref.vrt")

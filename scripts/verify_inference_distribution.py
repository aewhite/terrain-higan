"""Install an inference wheel in a clean environment and exercise bundled assets."""

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import venv

SMOKE_CODE = """
import importlib.util
import torch
from terrain_higan.inference import TerrainWorld, InferenceConfig
from terrain_higan.inference.weights import default_weights_dir

torch.set_num_threads(1)
assert importlib.util.find_spec('terrain_higan.projects') is None
assert importlib.util.find_spec('terrain_higan.etopo') is None
assert 'site-packages' in str(default_weights_dir())
world = TerrainWorld.from_default_weights(seed=1234, config=InferenceConfig(device='cpu'))
for zoom in (0, 1, 10, 11, 15, 16, 17, 18, 19, 20):
    result = world.generate(zoom=zoom, x=-3, y=2, width=3, height=2)
    assert result.shape == (1, 1, 2, 3)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
print('Installed package: default weights and z0/z1/z10/z11/z15-z20 CPU generation passed')
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        parser.error("Expected a wheel file")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="terrain-higan-inference-verify-") as temporary:
        work = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(work / "venv")
        executable = work / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run([str(executable), "-m", "pip", "install", f"{wheel}[demo]"], cwd=work, env=environment, check=True)
        subprocess.run([str(executable), "-I", "-c", SMOKE_CODE], cwd=work, env=environment, check=True)
        subprocess.run([
            str(executable), "-I", "-m", "terrain_higan.demos.map_to_image",
            "--device", "cpu", "--width", "4", "--height", "3",
            "--tile-size", "4", "--z-level", "0", "--seed", "1234",
            "--output-dir", str(work / "images"),
        ], cwd=work, env=environment, check=True)
        if len(list((work / "images").glob("*.png"))) != 3:
            raise RuntimeError("Demo did not produce its three PNGs")
    print("Clean-environment installation, inference, and image demo passed.")


if __name__ == "__main__":
    main()

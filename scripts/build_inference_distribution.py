"""Stage and build terrain-higan-inference with its default weights and optional demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_PATHS = (
    "z0_generator/latest/generator.safetensors",
    *(f"zn_map_upscaler/z{zoom}/latest/upscaler.safetensors" for zoom in range(1, 11)),
    *(f"zn_map_upscaler/z{zoom}s/latest/upscaler.safetensors" for zoom in range(11, 21)),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_project(stage: Path, weights_dir: Path, license_file: Path | None = None) -> None:
    missing = [name for name in WEIGHT_PATHS if not (weights_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing default weights: " + ", ".join(missing))
    template = ROOT / "packaging" / "inference"
    shutil.copy2(template / "pyproject.toml", stage / "pyproject.toml")
    shutil.copy2(template / "README.md", stage / "README.md")
    if license_file is not None:
        shutil.copy2(license_file, stage / "LICENSE")
    package = stage / "src" / "terrain_higan"
    package.mkdir(parents=True)
    shutil.copy2(ROOT / "src/terrain_higan/__init__.py", package / "__init__.py")
    shutil.copytree(
        ROOT / "src/terrain_higan/inference", package / "inference",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights"),
    )
    models = package / "models"
    models.mkdir()
    for name in ("__init__.py", "builders.py", "upscaling.py"):
        shutil.copy2(ROOT / "src/terrain_higan/models" / name, models / name)
    demos = package / "demos"
    demos.mkdir()
    shutil.copy2(ROOT / "src/terrain_higan/demos/__init__.py", demos / "__init__.py")
    shutil.copytree(
        ROOT / "src/terrain_higan/demos/map_to_image", demos / "map_to_image",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    records = []
    for zoom, relative in enumerate(WEIGHT_PATHS):
        destination = package / "inference" / "weights" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(weights_dir / relative, destination)
        records.append({
            "zoom": zoom, "path": relative,
            "sha256": sha256(destination), "bytes": destination.stat().st_size,
        })
    manifest = {
        "format_version": 1,
        "architecture": "wfcai-map-inference-v1",
        "weights": records,
    }
    metadata = tomllib.loads((stage / "pyproject.toml").read_text())
    manifest["release_version"] = metadata["project"]["version"]
    (package / "inference/weights/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def inspect_wheel(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        prefix = "terrain_higan/inference/weights/"
        expected = {prefix + name for name in WEIGHT_PATHS}
        actual = {name for name in names if name.endswith(".safetensors")}
        if actual != expected:
            raise ValueError("Wheel does not contain exactly the twenty-one default checkpoints")
        manifest = json.loads(archive.read(prefix + "manifest.json"))
        for record in manifest["weights"]:
            content = archive.read(prefix + record["path"])
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError(f"Wheel checksum mismatch: {record['path']}")
        for name in names:
            if name.startswith("terrain_higan/") and not (
                name == "terrain_higan/__init__.py" or name == "terrain_higan/demos/__init__.py"
                or name.startswith("terrain_higan/inference/")
                or name in {"terrain_higan/models/__init__.py", "terrain_higan/models/builders.py", "terrain_higan/models/upscaling.py"}
                or name.startswith("terrain_higan/demos/map_to_image/")
            ):
                raise ValueError(f"Unexpected package file in wheel: {name}")
            if name.endswith((".pt2", ".onnx", ".pyc")):
                raise ValueError(f"Unexpected artifact in wheel: {name}")


def write_checksums(directory: Path, files: list[Path]) -> None:
    (directory / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in sorted(files))
    )


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / ".local/weights")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist/inference")
    parser.add_argument("--license-file", type=Path, help="Owner-selected code/weight terms to include")
    parser.add_argument("--verify", action="store_true", help="Install in a fresh venv and run CPU smoke tests (downloads dependencies)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_arguments()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="terrain-higan-inference-build-") as temporary:
        stage = Path(temporary) / "project"
        stage.mkdir()
        stage_project(stage, args.weights_dir.resolve(), args.license_file)
        # Default build produces an sdist, then builds the wheel FROM that sdist.
        subprocess.run([sys.executable, "-m", "build", str(stage)], check=True)
        archives = sorted((stage / "dist").iterdir())
        wheel = next(path for path in archives if path.suffix == ".whl")
        inspect_wheel(wheel)
        subprocess.run([sys.executable, "-m", "twine", "check", *map(str, archives)], check=True)
        if args.verify:
            subprocess.run([sys.executable, str(ROOT / "scripts/verify_inference_distribution.py"), str(wheel)], check=True)
        release_name = wheel.name.removesuffix("-py3-none-any.whl")
        destination = output / release_name
        if destination.exists():
            raise FileExistsError(f"Release already exists: {destination}; bump the version or choose another output directory")
        destination.mkdir()
        for path in archives:
            shutil.copy2(path, destination / path.name)
        shutil.copy2(stage / "README.md", destination / "README.md")
        if (stage / "LICENSE").exists():
            shutil.copy2(stage / "LICENSE", destination / "LICENSE")
        share_files = [destination / wheel.name, destination / "README.md"]
        if (destination / "LICENSE").exists():
            share_files.append(destination / "LICENSE")
        with zipfile.ZipFile(destination / f"{release_name}-share.zip", "w", zipfile.ZIP_DEFLATED) as share:
            for path in share_files:
                share.write(path, path.name)
            share.writestr("SHA256SUMS", "".join(f"{sha256(path)}  {path.name}\n" for path in share_files))
        write_checksums(destination, list(destination.iterdir()))
        print(f"Release written to {destination}")
        print("Clean-environment smoke test passed." if args.verify else "Clean-environment smoke test NOT run; use --verify before sharing.")


if __name__ == "__main__":
    main()

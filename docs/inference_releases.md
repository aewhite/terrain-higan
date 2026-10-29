# Building and Sharing terrain-higan-inference

The release distribution is named `terrain-higan-inference`; its Python API is
`terrain_higan.inference`. The release script stages inference code, shared model
definitions, the map-to-image demo, and bundled weights in a temporary build tree.

## Build a Release

Use a Python environment with pip. The build itself does not require PyTorch;
the optional clean-install test installs runtime dependencies into a disposable
environment. From the repository root:

```bash
python -m pip install -r packaging/inference/requirements-build.txt
python scripts/build_inference_distribution.py --verify
```

By default all twenty-one safetensors are taken from `.local/weights` under the
checkout root (`--weights-dir` overrides this), preserving
the `z0_generator/latest` and `zn_map_upscaler/zN/latest` layout plus
`zn_map_upscaler/z11s/latest` through `z20s/latest`. Refresh
that directory with `scripts/sync_latest_weights.sh`
before building if needed. Missing checkpoints fail the build.

For a different weight snapshot or destination:

```bash
python scripts/build_inference_distribution.py --weights-dir /path/to/weights --output-dir /path/to/releases --verify
```

The single release version lives in `packaging/inference/pyproject.toml`.
Increment it whenever code or bundled default weights change. Update the wheel
filename example in `packaging/inference/README.md` alongside version changes.
The script refuses to overwrite an existing version directory.

The build produces a source archive first and builds the wheel from that
archive.
It inspects the wheel's contents and hashes and runs `twine check`. `--verify`
additionally creates a fresh virtual environment outside the checkout,
installs the wheel with `[demo]`, removes `PYTHONPATH`, runs CPU inference at
z0/z1/z10/z11/z15/z16/z17/z18/z19/z20, and executes the small image demo. This test requires network
access to install dependencies and may download a large PyTorch distribution.
Without `--verify`, the build prints that installation verification was skipped.

Expected output for version 0.1.2:

```text
dist/inference/terrain_higan_inference-0.1.2/
    terrain_higan_inference-0.1.2-py3-none-any.whl
    terrain_higan_inference-0.1.2.tar.gz
    terrain_higan_inference-0.1.2-share.zip
    README.md
    SHA256SUMS
```

ONNX and PT2 files,
training code, geospatial packages, datasets, and training dependencies are
excluded. Only `torch`, `numpy`, and `safetensors` are required at runtime;
`[demo]` adds Matplotlib. The wheel includes demo code whether or not that extra
is installed. The extra selects its dependencies.

## Send to a Friend

Send the `-share.zip`, which contains the wheel, recipient README, and checksum
list. Recipients install the wheel following the bundled README. Dependencies
are installed from a package index; the ZIP is not a complete offline environment.

Use a fresh environment for `terrain-higan-inference`; do not co-install the development
`terrain-higan` distribution there because both own files in the same import package.

The recipient README is maintained in `packaging/inference/README.md` and also
becomes the package-index description. The demo's more specific documentation
remains in `src/terrain_higan/demos/map_to_image/README.md`.

To verify a previously built wheel independently:

```bash
python scripts/verify_inference_distribution.py dist/inference/terrain_higan_inference-0.1.2/terrain_higan_inference-0.1.2-py3-none-any.whl
```

## Publish Later

Before a public release, choose the code and weight licensing terms, document
model provenance, and supply those owner-selected terms via `--license-file`.
The file is included as `LICENSE` in the staged project and shared ZIP. Add
the corresponding license metadata and project URLs to the release pyproject.
Confirm the distribution name is available on the target index.

Build and verify once. Upload only the wheel and source archive from that
version directory; the sharing ZIP, README, and checksum list are not Python
distribution uploads. For TestPyPI:

```bash
python -m twine upload --repository testpypi dist/inference/terrain_higan_inference-0.1.2/terrain_higan_inference-0.1.2-py3-none-any.whl dist/inference/terrain_higan_inference-0.1.2/terrain_higan_inference-0.1.2.tar.gz
```

TestPyPI may not contain runtime dependencies. Install the downloaded test
wheel with normal PyPI dependency resolution in a fresh environment rather
than changing the global index or mixing indexes indiscriminately. Once
verified, upload those same two artifacts to PyPI:

```bash
python -m twine upload dist/inference/terrain_higan_inference-0.1.2/terrain_higan_inference-0.1.2-py3-none-any.whl dist/inference/terrain_higan_inference-0.1.2/terrain_higan_inference-0.1.2.tar.gz
```

Use index-managed credentials, never commit tokens. Future CI can invoke the
same script and publish using PyPI Trusted Publishing. Neither build nor
verification scripts publish anything themselves.

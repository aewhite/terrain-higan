# Local setup and configuration

Install the development package
into your chosen environment with `python -m pip install -e '.[test]'`.
Training requires a CUDA-capable PyTorch installation and GPU. Inference and
preprocessing also support CPU. The inference API defaults to CPU; the image demo
selects CUDA when available. See `pyproject.toml` and
[test setup](../tests/README.md) for dependencies and supported runtimes.

## Directories

Development defaults derive from the checkout containing the source files and
work from any working directory after an editable install:

| Setting                     | Default relative to project root                                         |
|-----------------------------|--------------------------------------------------------------------------|
| `DATA_DIR`                  | `data/`                                                                  |
| `RUNS_DIR`                  | `runs/`                                                                  |
| `MODELS_DIR`                | `<data>/models/`                                                         |
| `ETOPO_VRT`                 | `<data>/etopo2022_15s/etopo_2022_15s_bedpref.vrt`                        |
| `--weights-dir`             | `.local/weights/` in development; bundled weights in installed inference |
| Weight-sync source argument | `data/models/` under the checkout                                        |

Datasets are under `<data>/datasets/etopo2022/15s/`, generated training maps under
`<data>/maps/`, and PDF/statistics analysis under `<data>/analysis/`. Model files
use the `z0_generator/latest/` and `zn_map_upscaler/zN/latest/` layouts.
Raw data, weights, logs, generated browser assets, and local overrides are ignored
by Git. Supply the raw ETOPO VRT and its referenced rasters before preprocessing;
these large artifacts are not included in a checkout. VRT raster references must
also be valid on your machine.

Relative paths in `src/overrides.py` resolve against the project root, and `~`
expands to your home directory. Explicit CLI paths are relative to the calling
working directory.

## Personal overrides

From the project root:

```sh
cp local_config.example.py src/overrides.py
```

Training, preprocessing, analysis, and TensorBoard load `src/overrides.py`
automatically through `terrain_higan.paths`. Edit the settings you need:

```python
from pathlib import Path

DATA_DIR = Path("/path/to/datasets-and-models")
RUNS_DIR = Path("/path/to/persistent/runs")
TENSORBOARD_HOST = "127.0.0.1"
TENSORBOARD_PORT = 6006
```

`MODELS_DIR` and `ETOPO_VRT` can be overridden separately; otherwise they follow
`DATA_DIR`. Omit any setting to retain its default. Explicit function, config,
and CLI arguments take precedence over these defaults.

The file is Git-ignored and excluded from Python distributions. For remote runs,
deploy it alongside the rest of `src/`; `scripts/rsync_src_to_remote.sh` includes
it in source syncs. Restart Python processes after editing settings.

## Data and training

```sh
python -m terrain_higan.etopo.process_etopo --workers 2 --prefetch-factor 2
python -m terrain_higan.etopo.analyze_patches --device cpu --batch-size 256
python -m terrain_higan.projects.z0_generator.train --device cuda:0
python -m terrain_higan.projects.zn_map_upscaler.train --z-level 1 --device cuda:0
```

Preprocessing accepts `--vrt`, `--base-dir`, `--runs-dir`, `--device`, `--workers`,
`--prefetch-factor`, `--steps`, `--min-z`, and `--max-z`. The zoom range is inclusive
and defaults to z0 through z10. For z0 through z3 only, run
`python -m terrain_higan.etopo.process_etopo --min-z 0 --max-z 3`.
Zero workers runs in the main process. Defaults
are eight workers and prefetch eight, so reduce these for smaller machines.
Each worker defaults to a 64 MiB GDAL cache; exported `GDAL_CACHEMAX`,
`GDAL_BLOCK_CACHE_SIZE`, and `GDAL_DISABLE_READDIR_ON_OPEN` override the defaults.
Dataset maintenance scripts accept `--base-dir` and share the same dataset root.

The training shard reader requires exactly one DataLoader worker and uses CUDA
prefetching.
Z0 exposes `--generated-size` to control generated-field/batch memory and
`--num-steps`. Upscaler training exposes `--input-maps-per-batch`, `--patches-per-batch`,
`--input-map-loading-workers`, `--input-map-generation-batch-size`, `--num-steps`,
and profiling options. Map batch counts must satisfy config
divisibility constraints. `--input-maps-per-batch` controls both predecessor-map
and real-patch candidate counts (default 16,384). `--patches-per-batch` selects
512 examples from each population for generator and critic training and must
not exceed the candidate count. Defaults are two loading workers and separate
post-training map-generation batches of 512; smaller GPUs may require smaller values.
Analysis and map generation also expose batch-size and device options.

## Weights and standalone inference

With trained exports under the checkout's `data/models/`, refresh the local
inference/browser snapshot:

```sh
scripts/sync_latest_weights.sh
```

Weight syncing also refreshes the web demo and requires Node 22+ on `PATH`.
Check `node --version` in the shell running the script. The script checks this
before transferring weights; older Node versions lack the CLI APIs used by the demo.

The shell script accepts `TERRAIN_HIGAN_WEIGHTS_SOURCE` and
`TERRAIN_HIGAN_WEIGHTS_DIR` for the source and destination. Optional positional
arguments take precedence over environment variables, then checkout-relative
defaults apply. Relative arguments and environment paths use the calling working
directory. For example:

```sh
export TERRAIN_HIGAN_WEIGHTS_SOURCE=trainer:/path/to/project/data/models
export TERRAIN_HIGAN_WEIGHTS_DIR="$HOME/terrain-weights"
scripts/sync_latest_weights.sh
```

Personal shell exports can live in an ignored `.local/` shell file or your shell
profile. To sync source code to a remote checkout, use
`scripts/rsync_src_to_remote.sh src <remote-host> <remote-project-dir>`.
This deletes stale files under the specified remote `src/` target.

Python inference accepts an explicit `weights_dir` API argument. Image generation,
browser model syncing/serving, the web test runner, and release building accept
`--weights-dir`. For example:

```sh
python -m terrain_higan.demos.map_to_image --weights-dir ./my-weights
python scripts/build_inference_distribution.py --weights-dir ./my-weights
node src/terrain_higan/demos/web/sync-models.cjs --weights-dir ./my-weights
node src/terrain_higan/demos/web/serve.cjs --weights-dir ./my-weights
```

A standalone inference wheel uses its verified bundled weights by default.
Its default image output is `runs/map_to_image/` under the current working
directory. Development image output follows `<runs>/map_to_image/`.
See [inference releases](inference_releases.md) for building and clean-install checks.

## Services and browsers

TensorBoard uses the active Python interpreter and defaults to `127.0.0.1:6006`.
Set `TENSORBOARD_HOST` and `TENSORBOARD_PORT` in the Python override file, pass
explicit function arguments, or use the training CLI:

```sh
python -m terrain_higan.main --tensorboard-host 127.0.0.1 --tensorboard-port 6010
```

TensorBoard startup removes timestamped run directories containing event files
but missing `.tensorboard_valid`. It also prunes directories beyond the newest
50 timestamped runs when the directory and its immediate contents have not been
modified for at least seven days.

The demo defaults to `http://127.0.0.1:4173/src/terrain_higan/demos/web/`.
Pass a port argument to `serve.cjs`; set `HOST=0.0.0.0` for deliberate LAN use.
WebGPU needs localhost or HTTPS. Supply `TLS_CERT` and `TLS_KEY` paths for HTTPS,
keeping certificate keys outside the served checkout. Personal service-manager
and certificate installation details belong in local notes. Dotfiles/directories
such as `.local` and `.idea` are not served.

Browser checks and diagnostics use Playwright-managed Chromium by default:

```sh
cd src/terrain_higan/demos/web
npm ci
npx playwright install chromium
npm test
npm run test:all
```

Set `CHROME_PATH` to explicitly use an installed browser. `DEMO_URL` selects an
existing server; otherwise the browser test runner starts a loopback server on
`TEST_PORT` (default 4189). Real model/browser checks need all synchronized ONNX
exports; run `npm run sync-models` first. Diagnostics default to the local demo
URL and write screenshots under `runs/diagnostics/` beneath the checkout. Pass
`--runs-dir` to choose another diagnostics output root.

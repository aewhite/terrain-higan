# Terrain HiGAN Project Summary

## Overview

Terrain HiGAN (Hierarchical Infinite GAN) is an experimental PyTorch project for learning terrain and elevation generators from ETOPO2022 global elevation data. A coarse generator and successive neural upscalers synthesize terrain across zoom levels.

The repo is a script-driven proof of concept, not an application service. Paths default to this checkout's data, models, and runs. Training requires CUDA; inference and preprocessing also support CPU. See [environment setup](environment.md) for local overrides and prerequisites.

## Current Shape

- Source code lives under `src/terrain_higan/`.
- Project-specific model groups live under `src/terrain_higan/projects/`.
- Shared training, model, ETOPO, TensorBoard, and visualization helpers live in package subdirectories under `src/terrain_higan/`.
- Python-backed demos live under `src/terrain_higan/demos/`. The static web terrain explorer is in `src/terrain_higan/demos/web/`, with a 2D map and a separate S2 3D globe at `globe.html`. Its README documents controls, local serving, and browser checks.
- Generated data and run artifacts are expected outside `src/`, primarily under `data/`, `runs/`, and `models/`.
- `pyproject.toml` is the dependency manifest and package configuration.

## Core Ideas

- Terrain is represented as single-channel elevation tensors.
- Raw ETOPO values are stored in shard files as `float32` meters. The training loader applies the shared universal elevation normalization immediately after reading them.
- ETOPO2022 patch shards are organized by zoom and patch size:

```text
data/datasets/etopo2022/15s/z{zoom}/{patch_size}x{patch_size}/shard-*
```

- The shared zoom constants define patch sizes for `z0` through `z10`:
  - `z0`: `3x3`
  - `z1-z2`: `4x4`
  - `z3-z4`: `6x6`
  - `z5-z6`: `8x8`
  - `z7-z10`: `10x10`
- Model inputs and outputs use the universal normalization defined in `terrain_higan.etopo.normalization`; terrain visualization denormalizes them to raw meters.

## Main Components

### Entry Point

`src/terrain_higan/main.py` configures PyTorch print/autograd settings, starts detached TensorBoard for `runs`, and trains the upscaler chain:

```python
for z in range(1, 21):
    train_zn_map_upscaler(ZnUpscalerConfig(z_level=z))
```

Train the z0 generator first using its direct module entry point.

### ETOPO Data Loading

`src/terrain_higan/etopo/dataloader.py` contains the active reusable shard loader.

- `DEFAULT_ETOPO_BASE_DIR` points at `<data>/datasets/etopo2022/15s`, with `<data>` defaulting to this checkout's `data/` directory.
- `ETOPOShardIterableDataset` discovers shard files for one zoom level and patch size.
- `build_etopo_shard_dataloader()` wraps the iterable dataset in a `DataLoader` with `num_workers=1`, pinned memory, prefetching, and persistent workers.
- `PatchLoaderStage` in `src/terrain_higan/training/patch_loader_stage.py` moves normalized batches to CUDA on a separate stream and reshapes them to `(batch, 1, patch_size, patch_size)`.

### ETOPO Processing and Analysis

`src/terrain_higan/etopo/` contains one-off data scripts.

- `process_etopo.py` samples random LAEA-projected patches from an ETOPO2022 VRT, downsamples them into zoom levels `z0` through `z10`, writes binary shard files, and logs preview images to TensorBoard.
- `analyze_patches.py` scans physical z0-z10 shards and derived z11-z20 patches, producing per-level and pooled statistics, histogram PNGs, a PDF report, and patch-mean PDF tensors. It also estimates normalization constants for review while all normalization uses existing constants. Its shared patch-mean helpers are reused by generated input-map analysis and training.
- Whole-dataset charts are named `physical_dataset_*` (pooled selected physical levels) and `z10_source_dataset_*` (the separate full z10 scan). Histogram axes follow each plotted population; residual and patch-standard-deviation charts show percentages per bin, while patch means/PDFs and sampled patch ranges show density on linear axes.
- `shuffle_shards.py` and `fixup_zero_samples.py` are maintenance utilities for generated shards.

To reshuffle only z0 through z4, preview with
`python -m terrain_higan.etopo.shuffle_shards --z-levels 0 1 2 3 4 --dry-run`,
then omit `--dry-run` to rewrite those levels. Omitting `--z-levels` processes all
discovered levels, including z1. Each zoom/patch-size directory is shuffled
independently; `--base-dir` selects the dataset root and `--seed` sets the random seed.

`process_etopo.py` defaults to this source VRT under the data root:

```text
data/etopo2022_15s/etopo_2022_15s_bedpref.vrt
```

### Shared Training Loop

Training is organized as explicit stage pipelines.

- `src/terrain_higan/training/training_loop.py` defines `TrainingContext`, the `TrainingLoopStage` protocol, and `TrainingLoop`.
- `TrainingLoop` creates fresh step data each step, runs stages in order, and appends `ProgressIndicatorStage`.
- `src/terrain_higan/training/patch_loader_stage.py` is the shared CUDA prefetching data stage.
- `src/terrain_higan/training/progress_indicator_stage.py` prints step count, elapsed time, percent complete, and step rate.

### Shared Models and Visualization

- `src/terrain_higan/models/builders.py` provides simple `deep_linear()` and `deep_conv2d()` builders.
- `src/terrain_higan/models/responsive_clipper.py` provides the output clipping module used by current generators and discriminators.
- `src/terrain_higan/models/zn_training_constants.py` stores shared patch sizes, output ranges, patch statistics, and zoom-difference statistics.
- `src/terrain_higan/visualization/terrain.py` converts elevation tensors into terrain-colored RGB images.
- `src/terrain_higan/visualization/grids.py` builds TensorBoard image grids.
- `src/terrain_higan/tensorboard_utils/tensorboard_helpers.py` starts TensorBoard detached.

## Active Project Packages

### `terrain_higan.projects.z0_generator`

This is the coarse generator required before training the upscaler chain.

- `config.py` defines `Z0GeneratorConfig`.
- `models.py` defines:
  - `Z0Generator`: a convolutional valid-convolution field generator.
  - `Z0Discriminator`: an MLP discriminator with a score head for `3x3`, `2x2`, or `1x1` patches.
- `train.py` builds the generator and a `3x3` discriminator on CUDA, loading only real `z0` patches.
- `stages.py` implements GAN training and TensorBoard logging. The generator combines the real `z0` 3x3 adversarial loss with elevation distribution loss. The critic combines adversarial loss with an output-mean penalty.
- TensorBoard logs score histograms under `scores/critic_real` and `scores/critic_fake`, with separate `_mean` scalar tags. Elevation histograms use `elevations/real` and `elevations/generated`; total-loss histograms over each logging interval use `loss/generator_histogram` and `loss/critic_histogram`, distinct from scalar loss tags. Images use the upscaler names `images/output_maps`, `images/generated_patches`, and `images/real_patches`.
- Runs are written directly under `runs` in timestamped directories like:

```text
runs/z0_YYYYMMDD_HHMMSS
```

### `terrain_higan.projects.zn_map_upscaler`

Trains successive map upscalers using materialized predecessor maps and real ETOPO
patches, exports model checkpoints, and rebuilds the current level's map pool for
the next level. `config.py` defines `ZnUpscalerConfig`; `train.py` is the direct
training entry point. See the workflow below for loading and profiling details.

`ZnUpscalerModel` uses a single sequential network on nine coarse channels:
parent elevation, four sine/cosine features, and four noise channels. Its layer
spec is `128/3 64/3/2 shuffle/2 32/3 16/3`, followed by a linear 1×1 residual
head. Convolutions use `channels/kernel[/dilation]`; `shuffle/factor` inserts
pixel shuffle and updates the following layer's input channel count. All
convolution strides are 1 and padding is zero. Each output uses only the
containing parent cell plus four neighbors per axis (9×9), including all four
packed noise channels in that window. The output remains
`(2H - 16) × (2W - 16)`, preserving tiled inference. Checkpoint loading supports
the current layer spec; use Git history for earlier topologies.

`ZnUpscalerCritic` applies spectral normalization to its feature convolutions,
including the convolution inside its context-aware first layer. Its final 1×1
score head and zero-initialized context modulation linear layer remain unnormalized.

## Data and Artifacts

Default project-root-relative directories:

- `data/`: raw ETOPO source files, generated datasets, and analysis output when present.
- `data/datasets/etopo2022/15s/`: local relative shard path produced by `process_etopo.py`.
- `runs/`: TensorBoard logs and generated preview images.
- `data/models/`: durable model artifacts, with per-project timestamped runs and `latest` links.
- `.local/weights/`: inference/release/browser weight snapshot.
- `src/terrain_higan/demos/web/assets/models/`: local ONNX checkpoints required by the Web demo. They are generated artifacts and are intentionally ignored by Git.
- `docs/`: project documentation.
- `.local/plans/`: agent-created planning notes.

## Dependencies

`pyproject.toml` declares:

- Python `>=3.10`
- `affine`
- `gradio`
- `matplotlib`
- `mpmath`
- `numpy`
- `pyproj`
- `rasterio`
- `safetensors`
- `tensorboard`
- `torch`

CUDA is required by the active training pipeline, including `PatchLoaderStage`.

## Common Workflows

### Build Patch Data

After the raw ETOPO VRT exists at the path expected by `process_etopo.py`:

```bash
python -m terrain_higan.etopo.process_etopo
```

This writes shard files under `data/datasets/etopo2022/15s/` and logs previews under `runs/etopo_patches`.

### Analyze Datasets

```bash
python -m terrain_higan.etopo.analyze_patches
```

This reads `<data>/datasets/etopo2022/15s/` by default and writes `patch_analysis-old.pdf`, individual chart PNGs, and `z{level}_patch_mean_pdf.pt` under `<data>/analysis/`. Pooled physical statistics exclude synthetic levels; a full z10 source summary and sampled normalization estimates are reported separately. Candidate constants are printed for review and never applied automatically.

Downsampling metrics include each target patch's population standard deviation
after 2x2 average pooling and its ratio to the original patch's standard deviation.
z0 uses its existing 3x3-to-1x1 pooling convention. Downsampled standard deviations
include flat patches; ratios exclude original flat patches, whose denominator is
zero, and retain zero when pooling removes all variation. Standard deviations use
normalized elevation units; ratios are unitless. The PDF and `*_downsampling.png`
charts show the distributions and a full-scan two-dimensional histogram relating
original and downsampled standard deviations, with bounded memory use.
`patch_std_comparison_log.png` overlays mean original and downsampled standard
deviations across z-levels on a logarithmic vertical axis. The same comparison is
included in `trends_log.png`. Per-level `zN_nonflat_patch_stds_histogram.png` charts
and the first panel of `zN_downsampling.png` overlay both distributions on shared
logarithmic axes and common bins. These comparisons use the same original
non-flat patches for both series. Downsampled zeros are included in the means,
omitted from log axes, and noted on the charts.

Use `--base-dir` and `--output-dir` for other locations, `--z-levels 0 1` for selected physical levels, and `--device cpu --sample-size 1000 --batch-size 256` for a small CPU run. Selecting z10 also analyzes synthetic z11-z20. `--normalization-sample-count` and `--seed` control the independent z10 random-pixel estimate.

### Train the Generator and Upscalers

```bash
python -m terrain_higan.projects.z0_generator.train
```

Then train one upscaler level, or use `terrain_higan.main` for the complete chain:

```bash
python -m terrain_higan.projects.zn_map_upscaler.train --z-level 1
```

### Exercise z(n) Batch Production

Upscaler TensorBoard images include `images/critic_real_highest`,
`images/critic_real_lowest`, `images/critic_fake_highest`, and
`images/critic_fake_lowest` at each image-logging step. Each grid shows the
highest or lowest critic-scored patches from that step's training batch, ordered
from the selected extreme, with a score above each patch. Configure the number
per grid with `critic_extreme_patch_count` (default 16) or
`--critic-extreme-patch-count`; batches smaller than this show all available patches.

For CPU/CUDA profiling of the map-upscaler training loop, see
[`zn-map-upscaler-profiling.md`](zn-map-upscaler-profiling.md).

Generate predecessor maps on your selected device:

```bash
python -m terrain_higan.projects.zn_map_upscaler.generate_input_maps --z-level 0
```

Each training-loop step assembles `input_maps_per_batch` materialized `z(n-1)`
maps and loads the same number of candidate real ETOPO patches (default 16,384
per population). PDF selection reduces both to `patches_per_batch` examples
(default 512), shared by generator and critic training. These counts are
configured with `--input-maps-per-batch` and `--patches-per-batch`. The separate
`input_map_generation_batch_size` controls inference batches when materializing
maps after training (default 512 per forward call). A group of 2,048 maps is
generated in four calls of 512 and assembled on the CPU for file writing.

Input-map batch assembly is independent of the number of maps per file. One
reader advances through sorted files and consecutive samples, retaining its file/sample cursor across
batches and wrapping only after the whole pool has been consumed. A partially
consumed file stays on the training device until exhausted, so a batch boundary
does not cause another read or upload of that file. Existing map
files do not need rebuilding when the training batch size changes. NumPy only
opens the map files. On CUDA, one coordinator owns the sequential file/sample
cursor, while `input_map_loading_workers` CPU workers (default 2) fill distinct
pinned buffers in parallel. The coordinator consumes their results in file order
and uploads the next batch asynchronously on a separate CUDA stream, alternating
between two reusable buffer sets. The training stream waits only for its batch's ready
event, and host buffers are not reused until their previous transfer completes.
Memory is bounded to the current and prefetched batches plus a partially consumed
file, rather than the map pool.
Each file transfers to the training device before batch assembly,
crop-coordinate sampling, crop extraction, and mean calculation.
PDF selection and gathering the selected context crops also stay on the device,
without copying selected indices back to the CPU. Each generated patch is sampled
from its selected metric crop's doubled footprint. This transfers full maps
rather than only selected crops (64 MiB per default 16,384-map batch of float32
32x32 maps, plus any unused maps in a partially consumed file).

Before training begins, `train_zn_map_upscaler()` rebuilds the z(n) input-map
PDF from the existing z(n-1) materialized map pool. After the model is saved, it
closes the loaders and writer and releases training state before rebuilding the
current z(n) map pool from that checkpoint for the next level. Materialization
crops parent maps and noise to the context needed by the saved output: 24×24
inputs for 32×32 maps. Centered crops preserve pixel phase and the full noise
draw, so saved maps and random-number progression match full-context inference.

Training compares mean critic loss and mean generator adversarial loss over
adjacent, non-overlapping windows of `loss_lookback_steps`
(`--loss-lookback-steps`). It continues if critic loss rises OR generator
adversarial loss falls, and stops when neither improves after at least
`minimum_steps` (`--minimum-steps`). Decisions occur at full-window boundaries
once two windows exist. `num_steps` (`--num-steps`) sets the hard cap and must
cover the minimum for normal training. See `ZnUpscalerConfig` for current defaults.
Loss sums stay on the training device and only transfer to CPU at window
boundaries. Profiling completes its capture before applying the stopping decision.

Upscalers clip normalized residual predictions to [-5, 5] before applying sinh.
Training uses the responsive clipper so gradients can move saturated predictions
back into range; inference applies the same limits. The residual bounds loss is
the mean squared violation divided by the level's residual divisor squared, making
its weight independent of batch size, patch size, and residual units. PDF selection
floors evaluated densities at the dtype's machine epsilon before inversion.
Large-map TensorBoard previews render z0 and every upscaler in overlapping input
crops stitched into 64x64 output tiles, retaining the full-map noise coordinates.
Partial z0 edge tiles reuse preceding context to satisfy the exported generator's
48-cell minimum input size, then crop back to the requested output region.

### Sync Source to Remote

`scripts/rsync_src_to_remote.sh` syncs a local source directory to a remote project `src/` directory:

```bash
scripts/rsync_src_to_remote.sh src <remote-host> <remote-project-dir>
```

The script excludes Python cache files and uses `--delete` for the remote `src/` target.

## Current Caveats

- The repo is experimental and requires supplied data/checkpoints for training/demo workflows.
- Active training requires CUDA and will fail on CPU-only machines.
- Development roots are centralized in `terrain_higan.paths`; personal Python overrides load automatically from ignored `src/overrides.py`.
- Maintained tests live under `tests/`, grouped by subsystem. `tests/web/` separates unit, integration, ONNX model, and browser checks. See [`tests/README.md`](../tests/README.md) for setup, pytest markers, and npm commands.
- The dependency manifest exists in `pyproject.toml`, but exact package versions are not pinned.

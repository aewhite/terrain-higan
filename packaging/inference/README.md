# terrain-higan-inference

Generate seeded terrain as PyTorch tensors, or render maps with the included
optional demo. The package includes the default z0 generator, ten regular
upscalers, and the z11s through z20s super-level upscalers;
no training repository, dataset, or model download is needed after installation.

## Install the Shared Release

Requires Python 3.10 or newer. Install the release wheel in an environment
separate from the development `terrain-higan` distribution: both contain the
`terrain_higan` import package.

```bash
python -m pip install "./terrain_higan_inference-0.1.2-py3-none-any.whl[demo]"
```

Omit `[demo]` for tensor inference only; the extra adds Matplotlib for image
generation. CPU inference requires no NVIDIA GPU. For CUDA, install the
appropriate PyTorch build using https://pytorch.org/get-started/locally/.

## Make an Image

From any working directory:

```bash
python -m terrain_higan.demos.map_to_image --device cpu --width 64 --height 64 --z-level 0 --seed 1234
```

The demo prints output locations and writes PNGs, normalized `.npy` elevation
data, and JSON metadata. Start small on CPU; deeper levels and larger maps take
more time. Use `--help` for image palettes, output directory, negative origins,
and device options. A larger example:

```bash
python -m terrain_higan.demos.map_to_image --device cpu --width 256 --height 192 --z-level 19 --x -128 --y 64 --seed 1234
```

## Use the Tensor API

```python
import torch
from terrain_higan.inference import (
    InferenceConfig, TerrainWorld, denormalize_elevation, meters_per_pixel,
)

world = TerrainWorld.from_default_weights(
    seed=1234,
    config=InferenceConfig(device=torch.device("cpu"), tile_size=64),
)
terrain = world.generate(zoom=0, x=-32, y=16, width=64, height=48)
print(terrain.shape)  # (1, 1, 48, 64), CPU float32, normalized elevation
elevation_m = denormalize_elevation(terrain)
print(meters_per_pixel(19))  # 1.0
```

Zoom is any integer from 0 through 20 inclusive. Horizontal resolution is
512 meters per pixel at z10, 1 meter at z19, and 0.5 meters at z20. Origins
are pixel coordinates at the requested zoom: x increases right, y down.
Increasing zoom by one halves meters per pixel. z11 through z20 use their
corresponding super-level checkpoints with each level's residual statistics and
bilinear 2x interpolation (`align_corners=False`), matching the web demo.
The bilinear-minus-nearest interpolation delta is included in the total residual
before clipping against that level's residual bounds.

The same seed, checkpoint set, and coordinates use the same model sequence and
coordinate-addressed noise field as the web demo. PyTorch and ONNX backends are
not bit-identical, and rounding differences can grow at deep zoom levels. Child
maps need not downsample exactly to their parents. This is
procedural terrain, not a geographic elevation lookup.

`generate()` allocates the whole requested region. Use
`world.iter_tiles(Region(...))` from `terrain_higan.inference` for bounded tile requests
on larger maps. Cache-backed tiles are read-only by convention: use `.clone()`
before mutation. For alternative project checkpoints, use
`TerrainWorld.from_weights(seed=..., z0_path=..., upscaler_paths={1: ..., ...})`
or `from_default_weights(weights_dir=...)` with the original directory layout.
The image CLI also accepts `--weights-dir`; relative arguments use your current
working directory. Without an explicit directory, the installed package verifies
and uses its bundled weights.

## Release Contents

The bundled `weights/manifest.json` records the release version and the SHA-256
of each safetensors file. Packaged default files are verified when loaded.
`SHA256SUMS` covers the release files outside the ZIP; the ZIP also contains
its own README/wheel checksum list.

Code and weight licensing terms are supplied by the release owner. No open
source license is implied by this package. Before public distribution, the
maintainer should include the chosen code/weight terms and model provenance.

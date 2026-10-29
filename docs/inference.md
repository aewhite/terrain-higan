# Inference

`terrain_higan.inference` generates normalized elevation tensors with the current z0 generator and map upscalers. It is independent of the training modules at runtime.

```python
import torch
from terrain_higan.inference import InferenceConfig, TerrainWorld, denormalize_elevation

world = TerrainWorld.from_default_weights(
    seed=1234,
    config=InferenceConfig(device=torch.device("cpu"), tile_size=64, batch_size=4),
)
normalized = world.generate(zoom=19, x=0, y=0, width=64, height=64)
meters = denormalize_elevation(normalized)
```

Coordinates are integer pixel coordinates in an unbounded planar field. `x` increases to the right and `y` increases downward. A request always returns a CPU float32 tensor shaped `(batch, 1, height, width)` with normalized elevations. Use `denormalize_elevation()` for meters.

The horizontal resolution is 512 meters per pixel at z10, one meter per pixel at z19, and 0.5 meters per pixel at z20. z11 through z20 use their corresponding super-level checkpoints, matching the web demo. Super-level inference uses the z10 residual scale and clipping statistics used during training.

Release wheels load and verify their bundled default weights. In a source checkout, defaults are discovered from `.local/weights` under the project root. Pass `weights_dir=` to `TerrainWorld.from_default_weights()` or `--weights-dir` to the image CLI to choose another directory; relative arguments use the calling working directory. Use `TerrainWorld.from_weights()` to supply checkpoint paths individually. Tile results are cache-backed and read-only by convention; call `.clone()` before modifying one.

## Implementation and Reproducibility

`TerrainWorld` owns a fixed seed, model chain, configuration, and private in-memory
tile cache. Use one caller at a time and construct another world to change its
identity. There is no disk cache or shared worker. `clear_cache()` releases retained
tiles but preserves lifetime hit/miss/eviction totals. Cache byte accounting covers
retained tensor storage, not tensors that callers keep after eviction.

`_geometry.py` owns the negative-coordinate parent/crop math; `world.py` groups
compatible shapes and restores request order; `noise.py` owns coordinate-addressed
noise; `models.py` owns the forward formulas. `weights.py` verifies checkpoint
compatibility and packaged hashes. These modules do not import training code.

Intermediate tensors stay on the execution device. Completed public results are
returned to CPU; `generate_regions()` moves each completed batch back before
starting the next, bounding accelerator memory. `batch_size` controls that API's
forward batches; `iter_tiles()` remains serial to preserve its current numerical
behavior and tile ordering.

Matching seeds and coordinates guarantee the same integer noise addressing, not
bit-identical neural outputs across devices, providers, batch sizes, or request
shapes. Tiny rounding differences can grow through deep model chains. Record the
weights, runtime/device, region shape, tile layout, and batch size when reproducing
a render. Never assemble model parents from arbitrarily overlapping cached
requests of different shapes.

The web URL stores camera position in meters; Python regions use pixel coordinates.
The web also uses 256-pixel display tiles with colormap-dependent rendering halos,
whereas the Python demo defaults to 64-pixel inference tiles and shades the
assembled map. For numerical comparisons, supply identical parent/noise tensors
or identical padded region requests before comparing rendered images.

The focused suite lives in `tests/inference/`, `tests/packaging/`, and
`tests/demos/`. Run it with pytest in the configured project environment.
Geometry/noise/palette fixtures in `tests/fixtures/` are also consumed by the web
unit checks. Real-model/browser checks remain separate from weight-free tests.
See [the testing guide](../tests/README.md) for setup and commands.

See [Building and Sharing terrain-higan-inference](inference_releases.md) for the release
script, clean-install verification, sharing ZIP, and package-index publishing.

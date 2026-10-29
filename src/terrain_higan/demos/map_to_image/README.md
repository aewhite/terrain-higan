# Map To Image

Generate a rectangular region from the public inference package and save its
normalized elevation data plus greyscale, elevation, and realistic terrain PNGs.

For a shared `terrain-higan-inference` wheel, install it with the `[demo]` extra in a
fresh virtual environment; the release README contains the exact install
command. All default weights are included in the wheel. Maintainers can follow
`docs/inference_releases.md` in the repository to build and verify a release.

```bash
python -m terrain_higan.demos.map_to_image --width 1600 --height 900 --z-level 19
```

The demo uses CUDA when it is available and CPU otherwise. Use `--device` to
override that selection, for example `--device cpu`. Release wheels use bundled
weights; source checkouts use `.local/weights/` under the project root. Pass
`--weights-dir /path/to/weights` to select another directory.

`--x` and `--y` select the pixel origin in the infinite terrain field. `--z-level`
accepts z0 through z20. Higher levels reduce meters per pixel while preserving
the requested pixel dimensions.

```bash
python -m terrain_higan.demos.map_to_image --z-level 20 --x -500 --y 125 --width 801 --height 503 --colormaps realistic elevation
```

Omit `--seed` to use the current Unix time in milliseconds. Output names include
that seed encoded with digits, upper-case letters, and lower-case letters; use
the printed numeric seed with `--seed` to reproduce a region with the same weights,
device/runtime, tile size, and inference settings. Different execution providers
or request shapes can accumulate rounding differences at deep zoom levels.

## Code and Memory

`generate.py` resolves CLI options, assembles inference tiles into a normalized
NumPy memmap, writes images, and constructs metadata. `colormaps.py` accepts meter
arrays and returns uint8 RGB; it does not know about models or tiles. Invalid
external elevation values render black. Generation uses only the public
`terrain_higan.inference` API.

The memmap bounds terrain-assembly memory, but rendering still allocates full-image
elevation/shading/RGB arrays. Palettes are written one at a time; PNG writing is
not streaming. The elevation palette retains Matplotlib's terrain LUT and the
Python renderer retains its one-sided outer-edge gradients. These intentionally
remain distinct from the web renderer's interpolation and padded tile edges.

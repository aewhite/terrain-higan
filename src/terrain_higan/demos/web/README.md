# Terrain HiGAN Web Demo

## GitHub Pages

The workflow in [deploy-pages.yml](../../../../.github/workflows/deploy-pages.yml)
publishes the map at `https://aewhite.github.io/terrain-higan/` and the globe at
`https://aewhite.github.io/terrain-higan/globe.html`.
In the repository's **Settings → Pages**, select **GitHub Actions** as the source.
Push demo changes to `main`, or run **Deploy web demo to GitHub Pages** manually
from the Actions tab on `main`.

The build downloads the `onnx-weights-v1.tar.gz` release asset from the
`onnx-weights-v1` tag, verifies its SHA-256, installs the pinned browser runtime,
synchronizes all 21 ONNX models, and runs the web unit tests. It uploads only
HTML, JavaScript, CSS, and the generated model/runtime assets. The weights,
dependencies, and staged site remain generated artifacts outside Git.

To publish new weights, create another release archive with the same directory
layout, then update `ONNX_WEIGHTS_TAG` and `ONNX_WEIGHTS_SHA256` in the workflow.
The archive contains `z0_generator/latest/generator.onnx`,
`zn_map_upscaler/z1/latest/upscaler.onnx` through `z10`, and
`zn_map_upscaler/z11s/latest/upscaler.onnx` through `z20s`.

## Globe viewer

Open `/src/terrain_higan/demos/web/globe.html` on the same server for the 3D
viewer, or follow **3D globe** from the map. Both pages share the panel
design, palettes, seed controls, local ONNX exports, and worker runtime. The globe
generates an Earth-sized planet with its own seed identity; it does not reproduce
the infinite map's locations or real Earth's continents.

Run `npm ci` and `npm run sync-models` in this directory after updating. Installation
also copies pinned Three.js browser assets and their licenses into the
ignored runtime directory. Rendering requires WebGL2. Inference uses the
WASM/WebGPU provider policy: WASM through z15 and optional WebGPU at z16–z20.
The globe supports the full z0–z20 hierarchy.

- Pinch with two fingers to zoom and pan on touch screens. Lifting one finger
  resumes rotation; pinching does not place distance measurements.
- A blank shaded planet appears while the first coarse tiles load. Once those
  tiles cover the visible globe, the planet shows generated terrain immediately
  and refines it in place. The progress bar stays visible until full detail is ready.
- Drag to carry the grabbed surface location with the pointer. Movement naturally
  slows as ground resolution increases. Wheel zoom follows the pointer; `+` and
  `-` zoom at the center. Zoom stops at native z20 resolution, with clearance
  above the displayed terrain, water, or seabed. The minimum clearance depends on
  physical viewport resolution and local S2 sample spacing; the maximum
  altitude is five Earth radii.
- Click two surface points for a great-circle distance. Dragging does not place
  points. A third click starts again; Clear or `Escape` removes the measurement.
  Measurement arcs add segments as you zoom in so their chords stay above the surface.
- Terrain mode places water at sea level and colors it using generated bathymetry.
  Oceans use rich blue offshore and brighter coastal blues in shallow water.
  Elevation and Contours expose the seabed. Pointer elevation always reports the
  generated terrain height, including negative values below water.
- Stars fade into the background at orbital distances and disappear when zooming
  toward the surface.
- Advanced offers true-scale relief (default), 5× or 10× exaggeration. Seed,
  palette, camera position/up direction, and exaggeration survive URL reloads.
  Links between pages carry the seed and palette while each viewer keeps its own
  camera representation.
  Saved camera positions support the full latitude range from −90° to +90°.
- `Home` restores the orbital view, `R` randomizes, `F` hides the panel, and `?`
  opens help. The scale represents the center of the view. Statistics use a
  28×18 grid of visible reference-sphere rays, excluding sky, and sample the
  displayed terrain at those directions. They are sampled screen-view statistics,
  rather than exact global or area-weighted statistics.

The globe uses S2’s six quadratic cube faces with `16 * 2^level` samples per side
(S2 cell level `level + 4`). Nominal z0 sample spacing is about 576 km, compared
with the planar model’s 524 km. S2 cells vary in area and local sample spacing.
The projection follows [Google S2’s coordinate transforms](https://github.com/google/s2geometry/blob/master/src/s2/s2coords.h),
with the axes rotated to the renderer’s Y-up convention.

Globe lighting uses a local sun 45° above the tangent plane, from the upper left
relative to the view. Latitude and camera altitude do not change that local
elevation. Terrain heights and normals are not smoothed for lighting.
The flat viewer also computes hillshade directly from unblurred elevation samples.
Its height gradients use each zoom level's sample spacing in meters, so the same
physical slope receives the same lighting across zoom levels (512 m spacing at z10).

Model levels are independent of mesh subdivision. Canonical center samples have
cross-face halos, globe-specific Philox noise, and a fixed four-sample overlap
collar that blends model predictions in neighboring chart frames. Children use
the finalized parent values. Three-way corner continuation projects the extended cube chart onto its dominant
face. Boundary ownership is deterministic. S2 changes generated worlds for a
given seed; the versioned cache identity prevents reuse of old blocks.

Blending can smooth chart boundaries and change learned residual statistics;
it does not make the models spherical or guarantee geophysical realism. All
terrain comes from the existing models. The 128 MiB inference cache stores raw
and finalized blocks separately. Display capacity follows the tiles required by
the current physical viewport, with a minimum of 512 tiles and 128 spare entries;
eviction disposes GPU resources. Selection culls the S2 quadtree against the
viewport and horizon before requesting tiles; offscreen siblings are not requested
for display. The geometry-node LRU retains at most 4,096 nodes or four times the
current selection size, whichever is larger; eviction also releases child references.
Cached child bounds and projection samples keep camera updates separate
from inference. Geometry refines to the camera's selected spatial patches immediately,
even when only a coarse model tile is available. Each patch reprojects its nearest
available ancestor's elevations and shares that ancestor's native texture. Zooming
therefore concentrates polygons near the camera without waiting for neural inference
or displaying giant flat parent triangles. Individual completed tiles replace their
own patches without a four-sibling loading barrier.
Orbit mesh updates use a 4 ms work budget per frame, yielding during stitching and
vertex construction. The existing mesh set stays visible until its replacements
can be swapped together, preserving shared edges. Only one replacement set is
prepared at a time; camera changes cancel obsolete work and release its meshes.
Progress remains active until both inference and mesh preparation finish.
Only coarse roots and final visible model tiles are requested for display, center-first;
intermediate model dependencies remain the generator's responsibility. Offscreen
display requests release the queue immediately. Obsolete dependency chains stop
before launching more inference, and queued obsolete jobs are skipped. Running
inference can finish and cache its block; current consumers retry cancelled shared
claims while reusing completed blocks. Hidden cached
model tiles do not allocate meshes, and unchanged frames do not render again.
Preview patches choose 4–64 subdivisions using camera distance, spherical curvature,
source-height curvature, and actual S2 chart interpolation error, targeting a
quarter-physical-pixel geometric error. Native patches use at least 16 subdivisions
through z8, 32 through z14, and 64 through z20; nonlinear chart regions can refine
above those minima. Subdivisions always cover the camera-refined spatial footprint rather than
the coarse source's original area. Mesh construction caches halo points and shares static
UV/index layouts. Height lookups use face/depth indexes rather than sorting
the tile cache, and pointer readouts are coalesced to animation frames.
Shared edge polylines and canonical corner constraints align neighboring meshes
across face boundaries, subdivision counts, and model LODs. Finer boundary vertices
follow the coarser edge in 3D rather than independently projecting onto the sphere;
skirts provide secondary coverage. Metric-correct normals avoid artificial shading
seams on flat terrain. Refinement is screen-driven and limited to
z20. Refinement measures projected model-sample spacing in physical screen pixels,
including device pixel ratio, and refines only when samples exceed 1.5 pixels.
This tolerates near-pixel detail instead of forcing another model level when a
sample is just over one pixel, which would halve its spacing again.
The former 48-branch refinement cap is removed. Native center samples feed color
textures without the old half-pixel averaging; separate vertex samples preserve
shared mesh boundaries. Camera clipping adapts to surface clearance, and local
mesh coordinates preserve precision at the finest levels. Camera clearance also
accounts for the vertex-interpolated surface used by coarse previews. Interior
vertex samples reuse the native texture halo instead of gathering those model
samples a second time; canonical cross-face vertices retain their original path.

The initial coarse model dependencies and worker sessions can take time to load.
The State row uses short, stable `Loading`, `Ready`, and `Error` labels. Ready is
debounced; model passes do not rewrite the status. Hover Error for its full cause.
The page remains navigable, and failures do not display substitute terrain. Globe
source and locally copied renderer assets participate in the existing model/code
bundle hash and reload policy.

### Globe checks

With models installed:

```sh
npm test
npm run test:integration
npm run test:models
npm run test:browser
npm run test:all
```

The browser runner starts a temporary server. `DEMO_URL` selects an existing server.
Install the default browser with `npx playwright install chromium` after `npm ci`.
`CHROME_PATH` selects an installed browser instead. The default headless run explicitly
uses SwiftShader WebGL2; use `HEADED=1` to check actual desktop GPU rendering.
Numerical checks cover all face boundaries and corners, noise aliases, actual
z0–z20 inference, native sample preservation, seed differences, and regeneration
after cache clearing. Navigation tests cover pixel-based LOD, Retina selection,
pole precision, grab direction, altitude sensitivity, and terrain clearance. Browser
checks cover navigation, z20 zoom, pixel resolution, stable status, palettes, distance measurement, URL
reloads, Retina/polar views, and explicit inference failures. Reports and screenshots
go under `runs/globe-viewer/`. Software-rendered timing is not a desktop GPU frame
rate measurement.

LOD regression checks intentionally withhold all fine-model replies: they verify
deep-zoom spherical ray hits, camera-local polygon refinement, shared-texture
ownership, idle-frame suppression, and clearance above exaggerated coarse previews.
Lifecycle checks cancel active and queued inference, then verify that current
consumers recover shared work and reuse completed blocks. Navigation checks also
exercise bounded node retention and exact-pole camera restoration.
Seam regressions cover reversed face edges, mixed spatial/model LODs, three-way
corners, polar chart interpolation error, and radial normals on constant-height terrain.

The performance check isolates viewport selection, tile construction, and height
sampling from ONNX and WebGL. It reports CPU timings at 1280×900 and asserts cached
selection, viewport culling, mesh/normal agreement, and geometry picking.
Timings are diagnostic, not machine-dependent pass/fail thresholds.

Source boundaries: `globe-topology.js` defines coordinates/units/geodesics;
`globe-mesh.js` reconciles shared mesh edges and corners;
`globe-terrain.js` owns spherical model blocks and reconciliation;
`globe-worker.js` coordinates generation through the shared inference pool;
`globe-tile-controller.js` schedules adaptive tiles and their display cache;
`globe-renderer.js` owns Three.js meshes, picking, lighting, and overlays;
`globe-app.js` owns DOM, input, URLs, and readouts.

## 2D viewer

`index.html` is a static terrain explorer. Serve the repository root and visit
`/src/terrain_higan/demos/web/`; worker-based ONNX inference requires HTTP(S), not `file://`.

The viewer is intentionally viewport-first: it schedules only the `256 x 256` model
tiles intersecting the screen, at one model sample per physical screen pixel
(the canvas includes the display's device pixel ratio). Tiles are
requested center-first and drawn immediately as they arrive. A coordinator worker
owns a shared 128 MiB LRU of complete, grid-aligned `64 x 64` normalized elevation
blocks at every generated level. Each missing block has one pending promise;
overlapping requests await that producer instead of duplicating inference.
Blocks are published only after inference completes and are never mutated.
Arbitrary requests, parent inputs, and rendering halos are assembled from those
canonical blocks, so request shape and colormap cannot change stored elevations.
Seed, camera
position, zoom, and colormap are retained in the URL hash. Its noise field is the
same coordinate-addressed Philox algorithm used by `terrain_higan.inference`. Matching
seeds alone do not promise identical neural terrain across providers or other
inference implementations; small rounding differences can grow through repeated upscaling.

Desktop browsers default to four independent inference workers; phones and tablets
default to one to limit WASM and model-session memory. Mobile uses the smaller
CPU-only WASM build, 128px map tiles and sequential 32px inference patches in
both views. It retains at most four model sessions, releases older sessions,
disables per-session memory arenas, and bounds the generated-terrain cache to
16 MiB. Mobile also uses a generated WASM loader capped at a 256 MiB shared-memory
maximum (the upstream loader requests 4 GiB), and prepares globe dependencies
sequentially to avoid retaining a large queued dependency tree. The loader
transformation is checked against the pinned ONNX Runtime package. Smaller
inference windows preserve the model halo and pixel-shuffle phase. Use `?workers=1` to select
the lower-memory path explicitly, or `?workers=2` through `?workers=4` to override it.
The inference workers load ONNX Runtime Web from `assets/runtime/` and z0 through
z10 plus z11s through z20s exported checkpoints from `assets/models/`. The WebGPU
mode uses WASM for z0 through z15, then WebGPU for z16 through z20 when available.
This keeps the coarse parent terrain consistent across backends: tiny provider
rounding differences in the early models otherwise grow into kilometer-scale
differences after repeated upscaling. The WASM mode uses WASM at every level.
z11 through z20 use their corresponding checkpoints, trained on z10 patches with
each patch's deviations scaled around its preserved mean. Each export embeds its
own measured residual scale, residual bounds, and elevation bounds from
`terrain_higan.etopo.normalization.UPSCALE_STATS`; the browser supplies only `input_map`
and `noise` and renders the returned normalized elevation directly.
Super-level residual statistics compare the current-level transformed z10 target
with the pooled, nearest-upscaled previous-level transform (untransformed z10
for z11). All z1 through z20 generators construct each 2x2 child block directly
from its parent's horizontal and vertical slopes, preserving the parent's mean.
Slopes use central differences inside the image and one-sided differences at its
edges. They include the delta from nearest interpolation in the
total residual before balancing and clipping. Outputs add that clipped total
residual to the nearest-upscaled parent.
The network's terrain and sine/cosine input features use the nearest-upscaled
parent; the slope-based map serves as the base for its learned corrections.

Zoom changes reuse cached model tiles, with one filtered draw for fractional scales.
Small pans crop the existing native-pixel mosaic instead of rebuilding it, and
elevation statistics wait until interaction pauses. Mobile display density is
bounded to reduce canvas and geometry work during gestures. Incoming
tiles replace the preview at native resolution. A separate bounded display cache
retains prior zoom levels; a new seed or colormap never displays another world's
preview. Obsolete display requests release immediately; shared block producers
already in progress can finish.

Rendering progress appears above the distance scale while work is pending. It
uses short single-line labels (Runtime, Download, Check, Load, Generate), then
disappears when the view is ready. Full failure details remain in the panel title
and the sidebar State readout; an allocation failure is labeled Out of memory. The ruler fills the
scale panel; its label uses that full width.

The ⛶ button at the top left enters fullscreen on supported browsers and is
hidden when unavailable. On iPhone, Safari’s Share → Add to Home Screen can open
the viewer as a standalone app (enable Open as Web App if shown).

On phones and tablets, controls start collapsed. Use the top-right menu button
to expand or collapse them. Keyboard shortcuts and the experimental walking
controls are hidden on mobile in both portrait and landscape.

Press `F` to enter map-only focus mode, keeping the distance scale in the map's
bottom-right corner. Press `F` again or `Escape` to restore the controls. Use the
keyboard icon or `?` to toggle the shortcuts dialog; `Escape` also dismisses it.
Zoom with the mouse wheel, a two-finger pinch, or `+` / `-` keys. Drag with one
finger to pan; two fingers can pan while pinching. The 2D camera zooms continuously;
wheel and keyboard zoom ease between scales. Inference uses the next native model
level, with cached terrain scaled while new detail loads. Shared URLs retain fractional zoom.

New 2D views and `Home` choose the nearest zoom level whose viewport covers
Earth’s surface area (about 510 million km²), accounting for screen size and
pixel density. A zoom level in a shared URL takes precedence.

Click or tap two map points to measure their
straight horizontal distance. The dashed line previews the distance before the
second point is placed; a third click starts another measurement. Distances use
the terrain's world-meter coordinates and display meters or kilometers, ignoring
elevation. Endpoints stay anchored while panning, zooming, or changing colormaps.
Clicking and dragging pans without placing a measurement point. The clear button
or `Escape` removes the measurement (after exiting focus mode, if active).
Changing the seed also clears it.

The `Contours` colormap draws minor and major elevation lines over a land-and-water
base. Lines are detected directly from the unblurred elevation field.
Its interval adapts from 1,000 m at continental levels
down to 10 m at the deepest model levels.

The controls show the model elevation at the pointer in real time. They also show
the exact minimum, maximum, mean, and standard deviation across the current
viewport once its visible tiles are ready.

The runtime assets are intentionally separate rather than embedded in one HTML
file. `npm ci` copies the pinned ONNX Runtime browser files into `assets/runtime/`.
The ONNX checkpoints under `assets/models/` are generated local artifacts and are
ignored by Git. `scripts/sync_latest_weights.sh` updates `.local/weights/` and
automatically installs z0 through z10 and z11s through z20s into the demo,
refreshing its bundle manifest. An optional weights destination is also used as
the source for the demo update. The Node dependencies must be installed first
with `npm ci` here. To refresh the demo independently, run `npm run sync-models`;
pass `--weights-dir /path/to/weights` to syncing, serving, and test commands when
using a weights directory other than `.local/weights/`.
Run it again after changing application HTML, JavaScript, CSS, or the pinned
runtime dependencies to refresh the code hash in the bundle manifest.
The server and tests compare all installed model hashes with the current exports
and refuse missing or stale files. `npm run check-models` runs that check alone.
The sync writes a bundle ID for the models and demo code after atomically replacing
changed ONNX files. The viewer shows its short ID and reloads when that ID changes, clearing model
sessions and elevation caches held by an older browser tab. The worker also
checks each ONNX file against the active bundle before opening a session.
Model bytes are fetched with `cache: "no-store"`; loaded sessions are reused in
each inference worker's memory. Each worker lazily opens only the models required
by the current zoom level while the page shell remains responsive.

Only model output is rendered. If WebGPU initialization fails, the worker retries
with the WASM CPU backend. If inference still fails, the actual error remains
visible and no procedural terrain is substituted.
All workers must select the same provider for each model; a mismatch is reported
as an inference error.

The main metrics show render state, visible tile progress, and elevation statistics.
The title's indicator is yellow while rendering and green when the view is ready;
failures turn it red and show the error in the State row. Both viewers show the
world seed in the main panel. Without a seed in the URL, the current timestamp in
milliseconds becomes the seed; clearing the seed field also selects a new timestamp.
The selected seed is saved in the URL for reproducible links and reloads.
Advanced contains the Compute provider, Threads readout, current Model, and weights bundle.
Threads shows the worker count and one CPU thread per worker.
Each inference worker uses one WASM thread. The pinned runtime's threaded convolutions round
differently for different region shapes; repeated upscaling can amplify this into
visible coastline breaks between neighboring tiles. The `threads` query parameter
is ignored. Desktop browsers default to four workers (the maximum); phones and
tablets default to one. `?workers=1` or `?workers=2` explicitly selects the pool size.
Each runtime owns its model sessions and WASM memory, while elevation blocks and
pending claims remain in the coordinator. Parent dependencies
are assembled before dispatching inference, so workers never wait for other jobs
in the pool. Only assembled input buffers are transferred to inference workers;
canonical cache buffers remain owned by the coordinator. Cancelled display requests
return promptly; the current bounded batch of block producers can finish and warm
the shared cache for subsequent requests.
Hover over Compute for the provider split or GPU fallback reason.
For diagnosis, `?backend=wasm` forces CPU inference at every level.

Start the local server from the repository root:

```sh
node src/terrain_higan/demos/web/serve.cjs
```

Open http://127.0.0.1:4173/src/terrain_higan/demos/web/ locally.
The server binds loopback by default; set `HOST=0.0.0.0` for access from another device.
Pass another port as an argument if needed, for example
`node src/terrain_higan/demos/web/serve.cjs 4174`. This server supplies JavaScript MIME types for
both `.js` and `.mjs`, which the ONNX WASM module requires.
It also supplies `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`. Other static hosts must send these
headers for cross-origin isolation. Each inference worker still uses one CPU thread.

### HTTPS on the LAN

WebGPU requires HTTPS (or localhost). The server accepts `TLS_CERT` and `TLS_KEY`
environment variables pointing to PEM files; keep private keys outside the repository,
since the HTTP server serves the repository root.
The server blocks dotfiles and the private `src/overrides.py` configuration.

```sh
HOST=0.0.0.0 TLS_CERT="/path/to/server.pem" TLS_KEY="/path/to/server-key.pem" \
  node src/terrain_higan/demos/web/serve.cjs 4177
```

Open the HTTPS URL using your machine's LAN hostname or address. The certificate
must cover that hostname/address, and each client must trust its issuing CA.
Transfer only the public CA certificate to clients; keep CA/server private keys
outside the checkout. See [environment setup](../../../../docs/environment.md)
for configuration.

Use a WebGPU-capable browser such as current Chrome. Certificate warnings must be
resolved rather than bypassed for normal use.

Developer checks (Node 22+; dependencies are only for development):

```sh
cd src/terrain_higan/demos/web
npm ci
npm test
npm run sync-models
npm run test:models
npm run test:browser
SEAM_ONLY=1 npm run test:browser -- test-browser.cjs
npm run format:check
```

Tests live under [`tests/web/`](../../../../tests/web/) and are documented in the
[testing guide](../../../../tests/README.md). The runner starts and stops the browser
test server automatically unless `DEMO_URL` is set. Browser tests check native Retina
dimensions, four independent single-thread runtimes, visible terrain during zoom,
cached zoom-out, and consistency between neighboring coastline tiles and a
continuous request. `SEAM_ONLY=1` checks worker-count invariance, halo/colormap
invariance, cached crops, cancellation, and the coastline regression, and reports
one-worker and four-worker cold and warm timings. Large Retina views can take longer to
generate than small views.
The measurement check verifies mouse and touch
distances at device pixel ratios 1 and 2, meter/kilometer units, anchored endpoints,
navigation, clearing, and seed changes. Its screenshots go under `runs/demo-measurement/`.
Set `CHROME_PATH` or `DEMO_URL` to override the local Chrome path or server URL.
The worker test checks canonical block reuse, history-independent output, and the
z10 and z11s through z20s residual formula and each level's measured scale and
bounds using the bundled ONNX models, plus z20 routing.
On a WebGPU-capable Mac, `MAX_ZOOM=20 node diagnose-backends.cjs` compares raw
elevations from headed Chrome's WebGPU and WASM workers at each zoom level.
`MODE=warm WARM_ZOOM=0 WARM_SIZE=32 WORLD_X=-293988345 WORLD_Y=31759354 node diagnose-backends.cjs`
checks that a z20 tile is identical after warming the cache in a separate worker.

## Source Boundaries

- `index.html` bootstraps bundle-versioned scripts; `styles.css` contains presentation.
- `app.js` owns DOM/camera/URL wiring, drawing, and readouts.
- `tile-controller.js` owns the display cache, visible tile queue, and single active
  request. IDs identify requests even when navigation returns to the same tile key.
- `model-worker.js` coordinates the shared cache, validates tagged requests, and
  composes generation/rendering. Cancellation releases the display request while
  bounded shared block production finishes. Obsolete replies cannot
  release a newer request or become current display tiles.
- `inference-pool.js` queues ready inference jobs across independent runtimes.
  `inference-worker.js` owns each runtime's execution and transferable inputs/outputs.
- `model-runtime.js` owns model verification, sessions, provider selection, and
  timing inside each inference worker. Each tile carries its resolved model and provider.
- `terrain.js` owns block recursion, pending claims, canonical region assembly,
  and the shared block LRU. Stored buffers are owned copies; messages transfer
  separate assembled or cropped buffers.
- `noise.js`, `rendering.js`, and `terrain-common.js` contain the numerical
  functions, palette/shading code, and shared region/unit/statistics helpers.

These are scoped classic scripts with explicit APIs on `TerrainHiGAN`, loaded in
entry-point order. The worker remains a classic worker because the pinned ORT
integration uses `importScripts`. There is no bundler or server inference.
The Node server serves files only; no cache is shared between browser tabs.

Browser requests use safe integer pixel coordinates within +/-2^50 (with room
for their extents and halos). URL camera coordinates are meters. Realistic and
elevation rendering and contours use one-pixel halos. These are assembled from
the same canonical blocks and do not change raw terrain. Cache identity includes
seed, zoom, and block coordinates; the bundle is fixed for the coordinator's lifetime.

`npm test` and `test:unit` need no model files and check pure functions, cache ownership,
request lifecycles, and bundle source coverage. `test:models` checks the current
ONNX models with the Node CPU runtime; this is not a WebGPU test. `test:all` runs
every group. The Playwright
suite checks the real WASM worker, previews, errors, cancellation/navigation,
desktop/mobile canvases, and bundle reloads. Screenshots go under
`runs/demo-inference-refactor/`. The headed diagnostic checks actual WebGPU
availability and reports the provider that produced each result.

Code/model bundle updates show a Reload notice instead of automatically resetting
the viewer. Reload when convenient to use the new bundle.

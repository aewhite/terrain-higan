# Terrain HiGAN

Terrain HiGAN is an experiment in generating landscapes based off of real world data using ML methods inspired by GANs and WFC.
It learns elevation patterns from Earth's ETOPO2022 data, then uses them to create
new terrain: continents, coastlines, mountains, and ocean floors.

The "Hi" in HiGAN means **Hierarchical Infinite**. Basically, the model attempts to refine existing inputs through a chain of upscalers that works on an infinite plane. The model can also work on finite fields such as spheres with some clever mapping.

The project includes an infinite map explorer, a 3D globe viewer, and tools for generating heightfields
and images in Python as well as the original training algorithms.

<table>
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="gallery/Screenshot%202026-10-27%20at%207.19.56%E2%80%AFAM.png"><img src="gallery/Screenshot%202026-10-27%20at%207.19.56%E2%80%AFAM.png" alt="Generated planet showing continents and ocean-floor relief" width="100%"></a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="gallery/Screenshot%202026-10-27%20at%207.20.09%E2%80%AFAM.png"><img src="gallery/Screenshot%202026-10-27%20at%207.20.09%E2%80%AFAM.png" alt="Generated globe with islands and continents surrounded by blue oceans" width="100%"></a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="gallery/Screenshot%202026-10-27%20at%207.45.31%E2%80%AFAM.png"><img src="gallery/Screenshot%202026-10-27%20at%207.45.31%E2%80%AFAM.png" alt="Generated globe with a broad central continent, surrounding islands, and underwater relief" width="100%"></a>
    </td>
  </tr>
  <tr>
    <td width="33%" align="center" valign="top">
      <a href="gallery/Screenshot%202026-10-27%20at%207.21.50%E2%80%AFAM.png"><img src="gallery/Screenshot%202026-10-27%20at%207.21.50%E2%80%AFAM.png" alt="Broad terrain view showing continental highlands and underwater relief" width="100%"></a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="gallery/Screenshot%202026-10-27%20at%207.22.24%E2%80%AFAM.png"><img src="gallery/Screenshot%202026-10-27%20at%207.22.24%E2%80%AFAM.png" alt="Generated continent with mountain ranges, coastal islands, and inland water" width="100%"></a>
    </td>
    <td width="33%" align="center" valign="top">
      <a href="gallery/Screenshot%202026-10-27%20at%207.25.04%E2%80%AFAM.png"><img src="gallery/Screenshot%202026-10-27%20at%207.25.04%E2%80%AFAM.png" alt="Close terrain view with winding coastlines, bays, and mountain ridges" width="100%"></a>
    </td>
  </tr>
</table>

## How it works

Rather than generating a finished landscape in one pass, Terrain HiGAN builds
heightfields at a series of scales:

1. A convolutional **coarse generator** turns noise into the initial elevation field.
2. A chain of **2× neural upscalers** adds detail, using the parent terrain and
   fresh noise to predict elevation changes at each level.
3. **GAN training** compares generated patches with elevation data to learn the
   terrain patterns at each scale.

Each level halves the horizontal sample spacing. In the planar world, the
hierarchy looks like this:

| Levels | Approx. sample spacing | Terrain features |
| --- | --- | --- |
| **z0** | 524 km | Planetery scale features |
| **z1–z10** | 262 km → 512 m | Continents to regions regional scale features |
| **z11-z20** | 256 m → 0.5 m | Local fine details |

The z11+ levels are currently **derived via super scaling**, reusing z10 terrain
patterns with reduced elevation variation. Their finer sample spacing represents
generated detail, not measured terrain.

## Explore the demos

The **[web demo](src/terrain_higan/demos/web/README.md)** runs neural inference
locally in the browser using ONNX models. It offers:

- An **infinite 2D map** to pan and zoom through generated terrain.
- An **Earth-sized 3D globe** with its own generated geography.
- Terrain, elevation, and contour views, seed controls, and distance measurements.

The **[map-to-image demo](src/terrain_higan/demos/map_to_image/README.md)** generates
a rectangular region and saves elevation data alongside greyscale, elevation,
and terrain-colored PNGs. The **[Python inference API](docs/inference.md)** lets
you request heightfields directly for your own experiments.

## Working with the project

The models and training code use PyTorch. Training requires CUDA; Python inference
and preprocessing also support CPU. Running the demos requires model checkpoints;
the demo guides cover setup.

For a deeper look:

- [Environment setup](docs/environment.md) — installation, data prerequisites, and local configuration.
- [Project guide](docs/summary.md) — code layout, data preparation, and training workflows.
- [Inference releases](docs/inference_releases.md) — building and sharing the standalone inference package.
- [Testing guide](tests/README.md) — Python and web checks.
- [AI policy](docs/ai-policy.md) — the project's position on AI and history of AI use.
- 
## Future goals

- Host the web demo on a public site so anyone can explore without local setup.
- Publish the models on Hugging Face to make them easier to download and use.
- Incorporate higher-resolution training data to learn terrain detail down to
  roughly 30 m resolution.
- Improve super-scaling for more convincing detail beyond the training data's
  resolution.
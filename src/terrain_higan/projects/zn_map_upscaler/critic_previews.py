import math

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from terrain_higan.visualization import create_image_grid


def make_scored_patch_grid(patches: torch.Tensor, scores: torch.Tensor) -> np.ndarray:
    """Render ranked terrain patches with critic scores above each tile (CHW RGB)."""
    patches = patches.detach().cpu()
    scores = scores.detach().cpu().flatten()
    count = patches.shape[0]
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    figure = Figure(figsize=(2 * columns, 2.3 * rows), dpi=100, layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False).flatten()
    for index, axis in enumerate(axes):
        axis.set_axis_off()
        if index < count:
            terrain = create_image_grid(
                patches[index:index + 1], upscale_factor=1, buffer_size=0,
            )
            axis.imshow(terrain.permute(1, 2, 0).numpy(), interpolation="nearest")
            axis.set_title(f"{scores[index].item():.6g}")
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy().transpose(2, 0, 1)

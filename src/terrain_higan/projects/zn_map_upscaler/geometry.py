"""Map geometry shared by materialization and training previews."""

import torch


def center_crop(maps: torch.Tensor, target_size: int) -> torch.Tensor:
    height, width = maps.shape[-2:]
    if height < target_size or width < target_size:
        raise ValueError(
            f"Cannot crop {tuple(maps.shape)} maps to {target_size}x{target_size}"
        )
    row_start = (height - target_size) // 2
    column_start = (width - target_size) // 2
    return maps[..., row_start:row_start + target_size, column_start:column_start + target_size]

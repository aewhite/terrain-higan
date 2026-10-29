from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from terrain_higan.etopo.dataloader import DEFAULT_ETOPO_BASE_DIR, build_etopo_shard_dataloader
from terrain_higan.training.training_loop import TrainingContext


class PatchLoaderStage:
    def __init__(
        self,
        *,
        z_level: int,
        patch_size: int,
        patches_per_batch: int,
        target_attr: str = "real_patches",
        base_dir: Path | str = DEFAULT_ETOPO_BASE_DIR,
        loop: bool = True,
        device: torch.device | str = "cuda",
    ) -> None:
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise ValueError(f"ETOPOShardDataStage requires a CUDA device, got {self._device}")

        self._patch_size = patch_size
        self._patches_per_batch = patches_per_batch
        self._target_attr = target_attr
        self._dataloader = build_etopo_shard_dataloader(
            base_dir=base_dir,
            z_level=z_level,
            patch_size=patch_size,
            patches_per_batch=patches_per_batch,
            loop=loop,
        )
        self._iterator: Any = iter(self._dataloader)
        self._queued_batch_count = 0
        self._finite_batch_count = None if loop else len(self._dataloader)
        self._prefetched_patches: torch.Tensor | None = None
        self._stream = torch.cuda.Stream(self._device)
        self._prefetch_next()

    @property
    def num_batches(self) -> int:
        return len(self._dataloader)

    def run(self, context: TrainingContext[Any]) -> None:
        if self._prefetched_patches is None:
            raise StopIteration("No prefetched ETOPO shard batch is available")

        real_patches = self._prefetched_patches
        self._prefetched_patches = None

        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_stream(self._stream)
        real_patches.record_stream(current_stream)

        setattr(context.step_data, self._target_attr, real_patches)
        self._prefetch_next()

    def _prefetch_next(self) -> None:
        if self._finite_batch_count is not None and self._queued_batch_count >= self._finite_batch_count:
            return

        cpu_batch = next(self._iterator).squeeze(0)
        self._queued_batch_count += 1

        with torch.cuda.stream(self._stream):
            real_patches = cpu_batch.to(self._device, non_blocking=True)
            real_patches = real_patches.reshape(
                self._patches_per_batch,
                1,
                self._patch_size,
                self._patch_size,
            )
            self._prefetched_patches = real_patches

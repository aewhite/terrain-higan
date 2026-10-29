from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from terrain_higan.etopo.normalization import normalize_elevation
from terrain_higan.paths import DEFAULT_DATA_DIR


DEFAULT_ETOPO_BASE_DIR = DEFAULT_DATA_DIR / "datasets" / "etopo2022" / "15s"


def _discover_shards(base_dir: Path | str, z_level: int, patch_size: int) -> list[Path]:
    data_dir = Path(base_dir) / f"z{z_level}" / f"{patch_size}x{patch_size}"
    shard_paths = sorted(data_dir.glob("shard-*"))
    if not shard_paths:
        raise FileNotFoundError(f"No ETOPO shards found in {data_dir}")
    return shard_paths


def iter_float32_batches(
    shard_paths: Sequence[Path],
    floats_per_batch: int,
    loop: bool = True,
) -> Iterator[torch.Tensor]:
    if floats_per_batch <= 0:
        raise ValueError(f"floats_per_batch must be positive, got {floats_per_batch}")
    if not shard_paths:
        raise ValueError("shard_paths must not be empty")

    bytes_per_float = 4
    bytes_per_batch = floats_per_batch * bytes_per_float
    shard_index = 0
    batch_bytes = bytearray()

    while True:
        if shard_index >= len(shard_paths):
            if not loop:
                return
            shard_index = 0

        with shard_paths[shard_index].open("rb") as shard_file:
            shard_index += 1

            while True:
                while len(batch_bytes) < bytes_per_batch:
                    chunk = shard_file.read(bytes_per_batch - len(batch_bytes))
                    if not chunk:
                        break
                    batch_bytes.extend(chunk)

                if len(batch_bytes) != bytes_per_batch:
                    break

                values = torch.frombuffer(memoryview(batch_bytes), dtype=torch.float32).clone()
                batch_bytes.clear()
                yield values

        if batch_bytes and shard_index >= len(shard_paths) and not loop:
            return


class LogicalFloatShardReader:
    def __init__(
        self,
        shard_paths: Sequence[Path],
        floats_per_sample: int,
        samples_per_read: int,
        loop: bool = True,
    ) -> None:
        if floats_per_sample <= 0:
            raise ValueError(f"floats_per_sample must be positive, got {floats_per_sample}")
        if samples_per_read <= 0:
            raise ValueError(f"samples_per_read must be positive, got {samples_per_read}")

        self._shard_paths = list(shard_paths)
        self._floats_per_sample = floats_per_sample
        self._samples_per_read = samples_per_read
        self._loop = loop

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from iter_float32_batches(
            self._shard_paths,
            self._floats_per_sample * self._samples_per_read,
            self._loop,
        )


class ETOPOShardIterableDataset(IterableDataset[torch.Tensor]):
    def __init__(
        self,
        base_dir: Path | str,
        z_level: int,
        patch_size: int,
        patches_per_batch: int,
        loop: bool = True,
    ) -> None:
        super().__init__()
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if patches_per_batch <= 0:
            raise ValueError(f"patches_per_batch must be positive, got {patches_per_batch}")

        self._shard_paths = _discover_shards(base_dir, z_level, patch_size)
        self._patch_size = patch_size
        self._patches_per_batch = patches_per_batch
        self._loop = loop
        self._complete_batch_count = self._count_complete_batches()

    def __len__(self) -> int:
        if self._loop:
            raise TypeError("Looping ETOPOShardIterableDataset does not have a finite length")
        return self._complete_batch_count

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = get_worker_info()
        if worker_info is not None and worker_info.num_workers != 1:
            raise RuntimeError("ETOPOShardIterableDataset expects DataLoader(num_workers=1)")

        reader = LogicalFloatShardReader(
            self._shard_paths,
            floats_per_sample=self._patch_size * self._patch_size,
            samples_per_read=self._patches_per_batch,
            loop=self._loop,
        )
        for batch in reader:
            yield normalize_elevation(batch)

    def _count_complete_batches(self) -> int:
        total_float_count = sum(path.stat().st_size // 4 for path in self._shard_paths)
        floats_per_batch = self._patches_per_batch * self._patch_size * self._patch_size
        return total_float_count // floats_per_batch


def build_etopo_shard_dataloader(
    *,
    base_dir: Path | str = DEFAULT_ETOPO_BASE_DIR,
    z_level: int,
    patch_size: int,
    patches_per_batch: int,
    loop: bool = True,
) -> DataLoader[torch.Tensor]:
    dataset = ETOPOShardIterableDataset(
        base_dir=base_dir,
        z_level=z_level,
        patch_size=patch_size,
        patches_per_batch=patches_per_batch,
        loop=loop,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
    )

#!/usr/bin/env python3
"""
One-off tool to shuffle patch order within each zN/pxp directory.

- Shuffles only within each zN/pxp directory (no cross zoom/patch-size).
- Keeps each patch's data unchanged; only the order of patches is randomized.
- Uses write-to-staging then rename-swap-drop to avoid corrupted writes.
- Loads at most 50 shard files (~500MB) per batch; chooses batch size so
  batches are as equal as possible (e.g. 102 files → batch size 34).

Usage:
    python -m terrain_higan.etopo.shuffle_shards --base-dir data/datasets/etopo2022/15s
    python -m terrain_higan.etopo.shuffle_shards --dry-run
    python -m terrain_higan.etopo.shuffle_shards --z-levels 0 1 2 3 4 --dry-run
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Reuse discovery from fixup_zero_samples
from terrain_higan.etopo.fixup_zero_samples import (
    discover_dataset_directories,
    find_shard_files,
    parse_patch_size_from_dir,
)

from terrain_higan.paths import DEFAULT_DATA_DIR

DEFAULT_BASE_DIR = str(DEFAULT_DATA_DIR / "datasets" / "etopo2022" / "15s")
MAX_FILES_PER_BATCH = 50
TARGET_SHARD_SIZE_MB = 10.0
TARGET_SHARD_SIZE_BYTES = int(TARGET_SHARD_SIZE_MB * 1024 * 1024)
STAGING_SUFFIX = ".tmp"
OLD_SUFFIX = ".old"


def batch_size_for_file_count(num_files: int) -> int:
    """
    Choose batch size so batches are as equal as possible and no batch exceeds 50 files.

    E.g. 102 files → 34 (102 = 34*3); 100 files → 50 (50*2).
    """
    if num_files <= 0:
        return 0
    if num_files <= MAX_FILES_PER_BATCH:
        return num_files
    num_batches = (num_files + MAX_FILES_PER_BATCH - 1) // MAX_FILES_PER_BATCH
    return (num_files + num_batches - 1) // num_batches


def load_batch_into_buffer(
    shard_paths: List[str],
    patch_elements: int,
) -> np.ndarray:
    """Load a list of shard files into one (num_patches, patch_elements) float32 buffer."""
    patch_bytes = patch_elements * 4
    chunks = []
    for path in shard_paths:
        size = os.path.getsize(path)
        if size % patch_bytes != 0:
            raise ValueError(
                f"Shard size {size} not divisible by patch size {patch_bytes}: {path}"
            )
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        n_patches = data.size // patch_elements
        chunks.append(data[: n_patches * patch_elements].reshape(-1, patch_elements))
    return np.concatenate(chunks, axis=0)


def write_shards_from_buffer(
    buffer: np.ndarray,
    patch_elements: int,
    staging_dir: Path,
    next_shard_index: int,
) -> int:
    """
    Write buffer (num_patches, patch_elements) to ~10MB shards under staging_dir.

    Shard filenames: shard-000000, shard-000001, ... starting at next_shard_index.
    Returns the next shard index after the last written shard.
    """
    patch_bytes = patch_elements * 4
    patches_per_shard = max(1, TARGET_SHARD_SIZE_BYTES // patch_bytes)
    staging_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    shard_index = next_shard_index
    while written < buffer.shape[0]:
        take = min(patches_per_shard, buffer.shape[0] - written)
        chunk = buffer[written : written + take]
        path = staging_dir / f"shard-{shard_index:06d}"
        with open(path, "wb") as f:
            f.write(chunk.astype(np.float32).tobytes())
        written += take
        shard_index += 1
    return shard_index


def shuffle_directory(
    dataset_dir: str,
    patch_size: int,
    dry_run: bool,
    seed: Optional[int],
) -> Tuple[int, int]:
    """
    Shuffle patches within one zN/pxp directory using staging and swap.

    Returns (total_patches, total_shards_written).
    """
    patch_elements = patch_size * patch_size
    patch_bytes = patch_elements * 4
    shard_files = find_shard_files(dataset_dir)
    if not shard_files:
        return 0, 0

    rng = random.Random(seed)
    rng.shuffle(shard_files)
    n_files = len(shard_files)
    batch_sz = batch_size_for_file_count(n_files)

    dir_path = Path(dataset_dir)
    staging_dir = dir_path.parent / (dir_path.name + STAGING_SUFFIX)
    old_dir = dir_path.parent / (dir_path.name + OLD_SUFFIX)

    if dry_run:
        total_patches = 0
        for path in shard_files:
            total_patches += os.path.getsize(path) // patch_bytes
        total_bytes = total_patches * patch_bytes
        num_output_shards = (total_bytes + TARGET_SHARD_SIZE_BYTES - 1) // TARGET_SHARD_SIZE_BYTES
        print(
            f"  [DRY RUN] {dataset_dir}: {n_files} files, batch_size={batch_sz}, "
            f"~{total_patches} patches → ~{num_output_shards} new shards"
        )
        return total_patches, num_output_shards

    staging_dir.mkdir(parents=True, exist_ok=True)
    next_shard_index = 0
    total_patches = 0

    for start in range(0, n_files, batch_sz):
        batch_paths = shard_files[start : start + batch_sz]
        buffer = load_batch_into_buffer(batch_paths, patch_elements)
        total_patches += buffer.shape[0]
        rng.shuffle(buffer)
        next_shard_index = write_shards_from_buffer(
            buffer, patch_elements, staging_dir, next_shard_index
        )
        del buffer

    # Swap: dir -> dir.old, staging -> dir, remove dir.old
    if old_dir.exists():
        shutil.rmtree(old_dir)
    dir_path.rename(old_dir)
    staging_dir.rename(dir_path)
    shutil.rmtree(old_dir)

    num_new_shards = next_shard_index
    print(f"  {dataset_dir}: {total_patches} patches → {num_new_shards} shards")
    return total_patches, num_new_shards


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Shuffle patch order within each zN/pxp directory (write-rename-swap-drop, batched)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base dataset directory (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--z-levels",
        type=int,
        nargs="+",
        default=None,
        help="Zoom levels to reshuffle (default: all discovered levels)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling (default: None)",
    )
    args = parser.parse_args(argv)

    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory does not exist: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    directories = discover_dataset_directories(args.base_dir)
    if args.z_levels is not None:
        selected_levels = {f"z{level}" for level in args.z_levels}
        directories = [
            (dir_path, patch_size)
            for dir_path, patch_size in directories
            if Path(dir_path).parent.name in selected_levels
        ]
        if not directories:
            parser.error("No dataset directories found for the requested --z-levels")
    if not directories:
        print(
            f"No dataset directories found in {args.base_dir}",
            file=sys.stderr,
        )
        print("Expected: {base_dir}/z{N}/{patch_size}x{patch_size}/shard-*", file=sys.stderr)
        sys.exit(1)

    total_patches = 0
    total_shards = 0
    for dir_path, patch_size in directories:
        p, s = shuffle_directory(dir_path, patch_size, args.dry_run, args.seed)
        total_patches += p
        total_shards += s

    if args.dry_run:
        print(f"\n[DRY RUN] Would process {len(directories)} directories")
    else:
        print(f"\nDone: {total_patches} patches across {total_shards} new shards in {len(directories)} directories")


if __name__ == "__main__":
    main()

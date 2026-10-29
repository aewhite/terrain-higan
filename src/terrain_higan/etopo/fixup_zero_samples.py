#!/usr/bin/env python3
"""
Utility to detect and remove all-zero samples from shard files.

Detects samples where every value is exactly 0.0, which indicates
a corrupted write (e.g., incomplete disk write, memory corruption).

This is a one-time fixup tool for cleaning up corrupted samples in the dataset.
Shard files are raw float32 binary files where each sample is patch_size * patch_size float32 values.

The script automatically discovers all zoom levels and patch size directories under the base path.
Patch size is derived from the directory name (e.g., "8x8" -> patch_size=8).

Usage:
    # Dry run (shows what would be removed without modifying files)
    python fixup_zero_samples.py --dry-run
    
    # Actually remove the samples
    python fixup_zero_samples.py --no-dry-run
    
    # Use a different base directory
    python fixup_zero_samples.py --base-dir /path/to/datasets --dry-run
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
from typing import List, Tuple, Optional


from terrain_higan.paths import DEFAULT_DATA_DIR


DEFAULT_BASE_DIR = str(DEFAULT_DATA_DIR / "datasets" / "etopo2022" / "15s")


def parse_patch_size_from_dir(dir_name: str) -> Optional[int]:
    """
    Parse patch size from directory name like '8x8', '16x16', etc.
    
    Returns:
        Patch size as int, or None if directory name doesn't match expected pattern.
    """
    match = re.match(r'^(\d+)x(\d+)$', dir_name)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        if width == height:
            return width
    return None


def find_shard_files(directory: str) -> List[str]:
    """Find all shard files in the given directory."""
    shard_pattern = os.path.join(directory, "shard-*")
    shard_files = sorted(glob.glob(shard_pattern))
    return shard_files


def discover_dataset_directories(base_dir: str) -> List[Tuple[str, int]]:
    """
    Discover all dataset directories under base_dir.
    
    Expects structure: base_dir/z{N}/{patch_size}x{patch_size}/shard-*
    
    Returns:
        List of (directory_path, patch_size) tuples
    """
    directories = []
    
    if not os.path.isdir(base_dir):
        return directories
    
    # Find all zoom level directories (z0, z1, z2, etc.)
    for zoom_dir_name in sorted(os.listdir(base_dir)):
        if not zoom_dir_name.startswith('z'):
            continue
        
        zoom_dir_path = os.path.join(base_dir, zoom_dir_name)
        if not os.path.isdir(zoom_dir_path):
            continue
        
        # Find all patch size directories within this zoom level
        for patch_dir_name in sorted(os.listdir(zoom_dir_path)):
            patch_size = parse_patch_size_from_dir(patch_dir_name)
            if patch_size is None:
                continue
            
            patch_dir_path = os.path.join(zoom_dir_path, patch_dir_name)
            if not os.path.isdir(patch_dir_path):
                continue
            
            # Check if there are shard files in this directory
            shard_files = find_shard_files(patch_dir_path)
            if shard_files:
                directories.append((patch_dir_path, patch_size))
    
    return directories


def detect_zero_samples(file_path: str, patch_elements: int) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Detect samples that are entirely 0.0 (indicating corrupted writes).
    
    Args:
        file_path: Path to the shard file
        patch_elements: Number of float32 elements per sample (patch_size * patch_size)
    
    Returns:
        Tuple of (data array, mask of valid samples, number of corrupted samples found, warning message or None)
    """
    patch_bytes = patch_elements * 4  # float32 is 4 bytes
    file_size = os.path.getsize(file_path)
    
    if file_size % patch_bytes != 0:
        return None, None, 0, f"Warning: File size {file_size} not divisible by patch size {patch_bytes}, skipping"
    
    num_samples = file_size // patch_bytes
    
    if num_samples == 0:
        return None, None, 0, "Warning: File is empty, skipping"
    
    # Read the entire file as flat array first
    with open(file_path, 'rb') as f:
        flat_data = np.frombuffer(f.read(), dtype=np.float32).copy()
    
    # Fast check: find all zero positions in the flat array
    zero_positions = np.where(flat_data == 0.0)[0]
    
    # If no zeros at all, we're done - all samples are valid
    if len(zero_positions) == 0:
        data = flat_data.reshape(num_samples, patch_elements)
        valid_mask = np.ones(num_samples, dtype=bool)
        return data, valid_mask, 0, None
    
    # Reshape for inspection
    data = flat_data.reshape(num_samples, patch_elements)
    
    # Find samples where ALL values are zero (entirely corrupted)
    all_zero_mask = np.all(data == 0.0, axis=1)
    valid_mask = ~all_zero_mask
    
    num_corrupted_samples = np.sum(all_zero_mask)
    
    return data, valid_mask, int(num_corrupted_samples), None


def process_file(file_path: str, patch_elements: int, dry_run: bool) -> Tuple[int, int]:
    """
    Process a single shard file, removing samples with corrupted writes.
    Prints messages immediately as issues are found.
    
    Args:
        file_path: Path to the shard file
        patch_elements: Number of float32 elements per sample
        dry_run: If True, don't modify the file
    
    Returns:
        Tuple of (original sample count, removed sample count)
    """
    data, valid_mask, num_corrupted, warning = detect_zero_samples(file_path, patch_elements)
    
    if warning:
        print(f"{file_path}: {warning}")
        return 0, 0
    
    if data is None:
        return 0, 0
    
    original_count = data.shape[0]
    
    if num_corrupted == 0:
        return original_count, 0
    
    # Found issues - print them immediately
    print(f"\n{file_path}:")
    print(f"  Found {num_corrupted} all-zero sample(s) out of {original_count} total")
    
    # Find indices of samples to be removed
    corrupted_indices = np.where(~valid_mask)[0]
    
    # Show sample indices being removed (up to 20)
    if len(corrupted_indices) <= 20:
        print(f"    Sample indices: {corrupted_indices.tolist()}")
    else:
        print(f"    Sample indices: {corrupted_indices[:20].tolist()} ... and {len(corrupted_indices) - 20} more")
    
    if dry_run:
        print(f"  [DRY RUN] Would remove {num_corrupted} samples, leaving {original_count - num_corrupted}")
    else:
        # Filter to keep only valid samples
        valid_data = data[valid_mask]
        
        # Write back to file
        with open(file_path, 'wb') as f:
            f.write(valid_data.astype(np.float32).tobytes())
        
        print(f"  Removed {num_corrupted} samples, {valid_data.shape[0]} samples remaining")
    
    return original_count, num_corrupted


def process_directory(dataset_dir: str, patch_size: int, dry_run: bool) -> Tuple[int, int, int, int]:
    """
    Process all shard files in a directory.
    Prints messages immediately as files are processed.
    
    Args:
        dataset_dir: Directory containing shard files
        patch_size: Patch size (derived from directory name)
        dry_run: If True, don't modify files
    
    Returns:
        Tuple of (files_processed, files_with_zeros, total_original, total_removed)
    """
    patch_elements = patch_size * patch_size
    
    shard_files = find_shard_files(dataset_dir)
    
    if not shard_files:
        return 0, 0, 0, 0
    
    total_original = 0
    total_removed = 0
    files_with_zeros = 0
    
    for shard_file in shard_files:
        original, removed = process_file(shard_file, patch_elements, dry_run)
        total_original += original
        total_removed += removed
        if removed > 0:
            files_with_zeros += 1
    
    return len(shard_files), files_with_zeros, total_original, total_removed


def main():
    parser = argparse.ArgumentParser(
        description="Detect and remove all-zero samples from shard files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--base-dir", "-b",
        type=str,
        default=DEFAULT_BASE_DIR,
        help=f"Base directory for datasets (default: {DEFAULT_BASE_DIR})"
    )
    
    parser.add_argument(
        "--dry-run", "-n",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show what would be removed without actually modifying files (use --no-dry-run to apply changes)"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.base_dir):
        print(f"Error: Base directory does not exist: {args.base_dir}")
        sys.exit(1)
    
    # Discover all dataset directories
    directories = discover_dataset_directories(args.base_dir)
    
    if not directories:
        print(f"No dataset directories found in {args.base_dir}")
        print("Expected structure: {base_dir}/z{N}/{patch_size}x{patch_size}/shard-*")
        sys.exit(1)
    
    # Process each directory (messages are printed immediately during processing)
    grand_total_files = 0
    grand_total_files_with_zeros = 0
    grand_total_original = 0
    grand_total_removed = 0
    
    for dir_path, patch_size in directories:
        files_processed, files_with_zeros, total_original, total_removed = process_directory(
            dir_path, patch_size, args.dry_run
        )
        grand_total_files += files_processed
        grand_total_files_with_zeros += files_with_zeros
        grand_total_original += total_original
        grand_total_removed += total_removed
    
    # Print summary if issues were found
    if grand_total_removed > 0:
        print(f"\n{'=' * 60}")
        print(f"Summary")
        print(f"{'=' * 60}")
        print(f"Base directory: {args.base_dir}")
        print(f"Detection: all-zero samples")
        print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE (will modify files)'}")
        print(f"Directories processed: {len(directories)}")
        print(f"Total files processed: {grand_total_files}")
        print(f"Total files with all-zero samples: {grand_total_files_with_zeros}")
        print(f"Total samples scanned: {grand_total_original}")
        print(f"Total all-zero samples {'to remove' if args.dry_run else 'removed'}: {grand_total_removed}")
        print(f"Total samples {'would remain' if args.dry_run else 'remaining'}: {grand_total_original - grand_total_removed}")
        
        if args.dry_run:
            print(f"\nRun with --no-dry-run to actually remove the all-zero samples.")


if __name__ == "__main__":
    main()

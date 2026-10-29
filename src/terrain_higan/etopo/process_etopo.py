import argparse
import itertools
import math
import random
import os
import json
import re
import sys
import gc
import shutil
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Any, Tuple, Union, Optional, BinaryIO
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import reproject
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import the shared colormap function
from terrain_higan.visualization import elevation_to_terrain_rgb
from terrain_higan.paths import DEFAULT_DATA_DIR, DEFAULT_ETOPO_VRT, DEFAULT_RUNS_DIR


ELEVATION_DATA = DEFAULT_ETOPO_VRT

# Worker-local storage for VRT file handles and call counter
# Each worker process will have its own copy of these variables
_worker_vrt = None
_worker_call_count = 0

def random_lat_lon() -> tuple[float, float]:
    u = random.uniform(-1, 1)
    lat0 = math.degrees(math.asin(u))
    lon0 = random.uniform(-180, 180)
    return lat0, lon0

def init_worker(worker_id: int, elevation_data: Path = DEFAULT_ETOPO_VRT):
    """
    Initialize worker process: open VRT once and set random seed.
    
    Args:
        worker_id: Unique ID for this worker (0-indexed)
    """
    global _worker_vrt, _worker_call_count, ELEVATION_DATA
    ELEVATION_DATA = elevation_data
    
    # Set a good random seed for this worker
    # Use worker_id and process ID to ensure different seeds per worker
    # Process ID ensures uniqueness even if multiple DataLoaders are used
    process_id = os.getpid()
    seed = (worker_id + 1) * 12345 + process_id
    random.seed(seed)
    np.random.seed(seed)
    
    # Limit GDAL cache to prevent memory growth
    # Set cache size to 64MB per worker (in MB, not bytes)
    os.environ.setdefault('GDAL_CACHEMAX', '64')
    # Reduce block cache size
    os.environ.setdefault('GDAL_BLOCK_CACHE_SIZE', '67108864')
    # Disable readdir on open to reduce memory
    os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    
    # Initialize call counter
    _worker_call_count = 0
    
    # Open VRT file once per worker
    _worker_vrt = rasterio.open(ELEVATION_DATA)
    
    print(f"Worker {worker_id} (PID {process_id}) initialized with seed {seed}")

def extract_patch(area_ratio: float = 0.005, output_size: int = 1536 * 2):
    global _worker_vrt, _worker_call_count
    gc.collect()
    
    lat, lon = random_lat_lon()

    # --- LAEA CRS centered at the random point (north-up at the center) ---
    crs_laea = CRS.from_string(f"+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

    # --- Define a square window that covers area_ratio of Earth's surface ---
    R = 6_371_000.0
    A = area_ratio * 4 * math.pi * R ** 2  # area_ratio of Earth's area (m²)
    s = math.sqrt(A)  # side of square (m)
    half_width = s / 2  # half-side
    res = s / output_size  # meters per pixel (derived from desired output size)
    W = H = output_size  # output image dimensions

    # LAEA grid transform: x in [-half_width, half_width], y in [-half_width, half_width] with (0,0) at the center
    # Affine(a,b,c, d,e,f): x = a*col + b*row + c; y = d*col + e*row + f
    transform = Affine(res, 0, -half_width, 0, -res, half_width)

    # --- Reproject the source into the target LAEA window (single band assumed) ---
    # Use the pre-opened VRT from worker initialization
    if _worker_vrt is None:
        # Fallback: open if not initialized (shouldn't happen in normal use)
        _worker_vrt = rasterio.open(ELEVATION_DATA)
    
    src = _worker_vrt
    # Pre-allocate array and reuse it to reduce memory allocations
    raw_patch = np.empty((H, W), dtype=np.float32)
    warp_extras = {"SAMPLE_GRID": "YES", "SOURCE_EXTRA": "100"}
    with WarpedVRT(
        src,
        crs=crs_laea,
        transform=transform,
        width=W,
        height=H,
        resampling=Resampling.bilinear,
        dtype="float32",
        warp_mem_limit=256,
        **warp_extras,
    ) as vrt:
        # Read directly into pre-allocated array to avoid extra copy
        vrt.read(1, out=raw_patch, out_dtype=np.float32)

    # Convert to tensor (torch.from_numpy shares memory, but DataLoader will copy when sending between processes)
    tensor = torch.from_numpy(raw_patch)
    
    return tensor


class ETOPOpatchDataset(Dataset):
    """Dataset that generates random ETOPO patches infinitely."""
    
    def __init__(self, area_ratio: float = 0.005, output_size: int = 1536):
        """
        Args:
            area_ratio: Ratio of Earth's surface area to cover
            output_size: Output patch size in pixels
        """
        # Use a very large size to make the dataset effectively infinite
        self.size = sys.maxsize
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Ignore idx since we generate random patches
        patch = extract_patch()
        # Convert to tensor and add channel dimension: (H, W) -> (1, H, W)
        patch_tensor = patch.unsqueeze(0)
        return patch_tensor

def build_sample(img: torch.Tensor, zoom_level: int, max_patches: int = 10000):
    """
    Build a sample from a batch tensor at a given zoom level.
    
    Args:
        img: Input tensor of shape (1, 1, 3072, 3072)
        zoom_level: Zoom level z from 0 to 11 inclusive
        max_patches: Maximum number of patches to extract (default: 100)
    
    Returns:
        Tensor of shape (P, 1, M*M) where:
        - P is the number of patches (all possible patches up to max_patches)
        - M is the patch size (z=0->3, z=1,2->4, z=3,4->6, z=5,6->8, z=7+->10)
        - Patches are flattened to M*M
    """
    assert img.shape == (1, 1, 3072, 3072), f"Expected shape (1, 1, 3072, 3072), got {img.shape}"
    assert 0 <= zoom_level <= 11, f"Zoom level must be 0-11, got {zoom_level}"
    
    # Remove batch and channel dimensions for processing: (3072, 3072)
    # img = img.squeeze(0).squeeze(0)  # (3072, 3072)
    N, C, H, W = img.shape
    
    # Compute grid size: (3 * 2^z) x (3 * 2^z)
    grid_size = 3 * (2 ** zoom_level)
    assert H % grid_size == 0, f"H {H} must be divisible by grid_size {grid_size}"
    
    # Patch sizes: z=0->3, z=1,2->4, z=3,4->6, z=5,6->8, z=7+->10
    if zoom_level == 0:
        M = 3
    elif zoom_level <= 2:
        M = 4
    elif zoom_level <= 4:
        M = 6
    elif zoom_level <= 6:
        M = 8
    else:
        M = 10
    K = H // grid_size
    downsampled = F.avg_pool2d(img, kernel_size=K)
    
    # downsampled shape: (N, C, grid_size, grid_size) = (1, 1, 3*2^z, 3*2^z)
    _, _, H_ds, W_ds = downsampled.shape
    
    # Check if patches fit within downsampled image
    if M > H_ds or M > W_ds:
        raise ValueError(f"Patch size M={M} exceeds downsampled image size {H_ds}x{W_ds}")
    
    # Calculate total number of possible patches without materializing
    num_patches_h = H_ds - M + 1
    num_patches_w = W_ds - M + 1
    total_patches = num_patches_h * num_patches_w
    
    # Determine how many patches to return
    P = min(total_patches, max_patches)
    
    # Determine which patches to extract
    if total_patches <= max_patches:
        # Extract all patches - generate all indices
        patch_indices = torch.arange(total_patches, device=downsampled.device)
    else:
        # Randomly sample max_patches indices without replacement
        patch_indices = torch.randperm(total_patches, device=downsampled.device)[:max_patches]
    
    # Convert linear indices to (row, col) coordinates
    top_indices = patch_indices // num_patches_w  # (P,)
    left_indices = patch_indices % num_patches_w   # (P,)
    
    # Create offset grids for patch extraction
    row_offsets = torch.arange(M, device=downsampled.device).view(1, M, 1)  # (1, M, 1)
    col_offsets = torch.arange(M, device=downsampled.device).view(1, 1, M)  # (1, 1, M)
    
    # Broadcast to get all patch coordinates: (P, M, M)
    h_idx = top_indices.view(P, 1, 1) + row_offsets  # (P, M, M)
    w_idx = left_indices.view(P, 1, 1) + col_offsets  # (P, M, M)
    
    # Extract patches directly from downsampled image: (P, M, M)
    # downsampled is (N, C, H_ds, W_ds), we need to index into spatial dims
    patches = downsampled[0, 0, h_idx, w_idx]  # (P, M, M)
    
    # Flatten patches: (P, M, M) -> (P, M*M)
    patches_flat = patches.view(P, M * M)  # (P, M*M)
    
    # Add channel dimension: (P, 1, M*M)
    return patches_flat

def extract_samples(batch: torch.Tensor, z_levels: range = range(11)):
    samples = []
    for z in z_levels:
        sample = build_sample(batch, zoom_level=z)
        samples.append(sample)
    return samples


@dataclass
class ShardState:
    """State for a single zoom level's shard file."""
    shard_num: int = 0
    file: Optional[BinaryIO] = None
    size: int = 0
    patch_size: int = 0


class ShardManager:
    """Manages shard files for writing samples, rotating when shards exceed size limit."""
    
    def __init__(self, base_dir: str | Path = DEFAULT_DATA_DIR / "datasets/etopo2022/15s", max_shard_size_mb: float = 10.0):
        """
        Args:
            base_dir: Base directory for shard files
            max_shard_size_mb: Maximum shard size in MB before rotating to next shard
        """
        self.base_dir = Path(base_dir)
        self.max_shard_size_bytes = int(max_shard_size_mb * 1024 * 1024)
        
        # Track shard state for each zoom level
        self.shard_state: Dict[int, ShardState] = {}
    
    def _get_shard_path(self, zoom: int, shard_num: int, patch_size: int) -> Path:
        """Get the path for a shard file."""
        return self.base_dir / f"z{zoom}" / f"{patch_size}x{patch_size}" / f"shard-{shard_num:06d}"
    
    def _find_last_shard(self, zoom: int, patch_size: int) -> int:
        """
        Find the last (highest numbered) shard file for the given zoom and patch size.
        Returns the shard number, or -1 if no shards exist.
        """
        shard_dir = self.base_dir / f"z{zoom}" / f"{patch_size}x{patch_size}"
        
        if not shard_dir.exists():
            return -1
        
        # Find all shard files matching the pattern
        shard_files = list(shard_dir.glob("shard-*"))
        if not shard_files:
            return -1
        
        # Extract shard numbers from filenames
        shard_nums = []
        for shard_file in shard_files:
            # Extract number from filename like "shard-000123"
            match = re.search(r'shard-(\d+)', shard_file.name)
            if match:
                shard_nums.append(int(match.group(1)))
        
        if not shard_nums:
            return -1
        
        return max(shard_nums)
    
    def _open_shard(self, zoom: int, patch_size: int, force_new: bool = False):
        """
        Open a shard file for the given zoom level and patch size.
        If force_new is False, will resume from the last existing shard.
        
        Args:
            zoom: Zoom level
            patch_size: Patch size M
            force_new: If True, always start a new shard. If False, resume from last shard.
        """
        # Close existing file if open
        if zoom in self.shard_state and self.shard_state[zoom].file is not None:
            self.shard_state[zoom].file.close()
        
        # Initialize state if needed
        if zoom not in self.shard_state:
            self.shard_state[zoom] = ShardState(
                shard_num=0,
                file=None,
                size=0,
                patch_size=patch_size
            )
        
        # Find the last shard if not forcing a new one
        if not force_new:
            last_shard_num = self._find_last_shard(zoom, patch_size)
            if last_shard_num >= 0:
                # Resume from the last shard (rotation will happen after writes if needed)
                self.shard_state[zoom].shard_num = last_shard_num
            # else: no existing shards, shard_num stays at 0
        
        # Get shard path and create directory
        shard_path = self._get_shard_path(zoom, self.shard_state[zoom].shard_num, patch_size)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get current file size if it exists
        current_size = shard_path.stat().st_size if shard_path.exists() else 0
        
        # Validate file size is divisible by patch size in bytes (M*M*sizeof(float32))
        if current_size > 0:
            patch_size_bytes = patch_size * patch_size * 4  # float32 is 4 bytes
            if current_size % patch_size_bytes != 0:
                # File is corrupted - move it to corrupted directory and start fresh
                corrupted_dir = self.base_dir.parent / "corrupted"
                corrupted_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a unique name for the corrupted file (include zoom, patch_size, and shard_num)
                corrupted_name = f"z{zoom}_{patch_size}x{patch_size}_shard-{self.shard_state[zoom].shard_num:06d}"
                corrupted_path = corrupted_dir / corrupted_name
                
                # Move the corrupted file
                shutil.move(str(shard_path), str(corrupted_path))
                print(f"WARNING: Corrupted shard file {shard_path} moved to {corrupted_path} "
                      f"(size {current_size} bytes not divisible by {patch_size_bytes})")
                
                # Reset size to 0 to start fresh
                current_size = 0
        
        # Open file in append mode (binary)
        shard_file = open(shard_path, 'ab')
        self.shard_state[zoom].file = shard_file
        self.shard_state[zoom].size = current_size
        self.shard_state[zoom].patch_size = patch_size
        
        print(f"Opened shard for zoom {zoom}: {shard_path} (current size: {self.shard_state[zoom].size} bytes)")
    
    def _check_and_rotate(self, zoom: int, patch_size: int):
        """Check if shard needs rotation and rotate if necessary."""
        if self.shard_state[zoom].size >= self.max_shard_size_bytes:
            # Rotate to next shard (force_new=True to create next sequential shard)
            self.shard_state[zoom].shard_num += 1
            self._open_shard(zoom, patch_size, force_new=True)
    
    def write_sample(self, zoom: int, sample: torch.Tensor):
        """
        Write an entire sample tensor to the appropriate shard file.
        
        Args:
            zoom: Zoom level (0-10)
            sample: Tensor of shape (N, M*M) with float values where N is number of patches
        """
        sample_np = sample.contiguous().detach().cpu().numpy().astype(np.float32)
        
        # Derive patch size M from tensor shape: M*M = sample.shape[1]
        M_squared = sample.shape[1]
        M = int(math.sqrt(M_squared))
        assert M * M == M_squared, f"Expected square patch size, got {M_squared} elements"

        # Verify patch_size consistency if state already exists
        if zoom in self.shard_state and self.shard_state[zoom].patch_size != 0:
            assert self.shard_state[zoom].patch_size == M, \
                f"Patch size changed for zoom {zoom}: expected {self.shard_state[zoom].patch_size}, got {M}"

        # Initialize or open shard if needed
        if zoom not in self.shard_state or self.shard_state[zoom].file is None:
            self._open_shard(zoom, M, force_new=False)  # Resume from last shard if exists

        # Write entire sample as one binary operation
        self.shard_state[zoom].file.write(sample_np.tobytes())
        self.shard_state[zoom].size += sample_np.nbytes
        
        # Check if we need to rotate shard after writing
        self._check_and_rotate(zoom, M)
    
    def flush_all(self):
        """Flush all open shard files to disk."""
        for zoom in self.shard_state:
            if self.shard_state[zoom].file is not None:
                self.shard_state[zoom].file.flush()
    
    def close_all(self):
        """Close all open shard files."""
        for zoom in self.shard_state:
            if self.shard_state[zoom].file is not None:
                self.shard_state[zoom].file.close()
                self.shard_state[zoom].file = None


def log_images_to_tensorboard(
    batch: torch.Tensor,
    samples: List[torch.Tensor],
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    z_levels: range = range(11),
) -> None:
    """
    Log images to TensorBoard: original downscaled image and first patch from each sample.
    
    Args:
        batch: Input batch tensor of shape (1, 1, H, W)
        samples: List of sample tensors, one for each zoom level, each of shape (P, M*M)
        writer: TensorBoard SummaryWriter instance
        step: Current training step
        device: Device to use for tensor operations
        z_levels: Zoom levels corresponding to the samples
    """
    # Log the original batch image, downscaled to 1024x1024
    batch_on_device = batch.to(device)
    # batch shape is (1, 1, H, W), extract the 2D image
    original_2d = batch_on_device.squeeze(0).squeeze(0)  # Shape: (H, W)
    # Downscale to 1024x1024 using interpolation
    original_downscaled = F.interpolate(
        original_2d.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims: (1, 1, H, W)
        size=(1024, 1024),
        mode='bilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # Remove dims: (1024, 1024)
    # Convert to RGB image using the existing color map
    original_rgb = elevation_to_terrain_rgb(original_downscaled)  # Shape: (3, 1024, 1024)
    # Log to TensorBoard
    writer.add_image("original/downscaled", original_rgb, step)
    
    # For each sample (zoom level), log the first patch as an image
    for z, sample in zip(z_levels, samples):
        # sample shape is (P, M*M) where P is number of patches, M is the patch size for zoom level z
        # Select the first row (first patch)
        first_patch = sample[0]  # Shape: (M*M,)
        
        # Calculate M from the shape
        M = int(math.sqrt(first_patch.shape[0]))
        assert M * M == first_patch.shape[0], f"Expected square patch, got {first_patch.shape[0]} elements"
        
        # Reshape to (M, M)
        patch_2d = first_patch.view(M, M)  # Shape: (M, M)
        
        # Convert to RGB image using the existing color map
        rgb_image = elevation_to_terrain_rgb(patch_2d)  # Shape: (3, M, M)
        
        # Log to TensorBoard
        writer.add_image(f"sample/z{z}", rgb_image, step)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample ETOPO patches into zoom-level shards")
    parser.add_argument("--vrt", type=Path, default=DEFAULT_ETOPO_VRT)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_DATA_DIR / "datasets/etopo2022/15s")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR / "etopo_patches")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--min-z", type=int, default=0, help="First zoom level to extract (inclusive, 0-10)")
    parser.add_argument("--max-z", type=int, default=10, help="Last zoom level to extract (inclusive, 0-10)")
    args = parser.parse_args(argv)
    if args.workers < 0 or args.prefetch_factor <= 0 or args.steps <= 0:
        parser.error("workers must be non-negative; prefetch-factor and steps must be positive")
    if not 0 <= args.min_z <= args.max_z <= 10:
        parser.error("zoom levels must satisfy 0 <= min-z <= max-z <= 10")
    return args


def main(argv: list[str] | None = None):
    args = parse_arguments(argv)
    z_levels = range(args.min_z, args.max_z + 1)
    # Setup device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Image logging interval (every S steps)
    image_log_interval_steps = 250
    
    # Create dataset and dataloader (infinite dataset)
    dataset = ETOPOpatchDataset()
    loader_options = (
        dict(
            persistent_workers=True,
            worker_init_fn=partial(init_worker, elevation_data=args.vrt),
            prefetch_factor=args.prefetch_factor,
        )
        if args.workers else {}
    )
    if not args.workers:
        init_worker(0, args.vrt)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        in_order=False,
        num_workers=args.workers,
        pin_memory=True if torch.cuda.is_available() else False,
        **loader_options,
    )
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=args.runs_dir)
    
    # Setup shard manager for writing samples to files
    shard_manager = ShardManager(base_dir=args.base_dir)
    
    dataloader_iterator = iter(dataloader)
    # Get a batch and move to GPU
    for step in range(args.steps):
        batch = next(dataloader_iterator)
        print(f"Step {step}: Batch shape: {batch.shape}")  # Should be (batch_size, 1, H, W)
        samples = extract_samples(batch.to(device), z_levels)
        
        # Write each sample to shard files
        for zoom, sample in zip(z_levels, samples):
            # sample shape is (P, M*M) where P is number of patches
            # Write entire sample as one operation
            shard_manager.write_sample(zoom, sample)
        
        # Flush all shard files to disk after every step
        shard_manager.flush_all()
        
        # Log images every S steps
        if step % image_log_interval_steps == 0:
            log_images_to_tensorboard(batch, samples, writer, step, device, z_levels)
        
    
    # Clean up
    shard_manager.close_all()
    writer.close()


if __name__ == "__main__":
    main()

"""Coordinate-addressed Philox noise fields."""

import hashlib

import numpy as np
import torch

_MASK32 = np.uint64(0xFFFFFFFF)
_PHILOX_M0 = np.uint64(0xD2511F53)
_PHILOX_M1 = np.uint64(0xCD9E8D57)
_PHILOX_W0 = np.uint32(0x9E3779B9)
_PHILOX_W1 = np.uint32(0xBB67AE85)


def _mulhilo(left: np.ndarray, right: np.uint64) -> tuple[np.ndarray, np.ndarray]:
    product = left.astype(np.uint64) * right
    return (product >> np.uint64(32)).astype(np.uint32), (product & _MASK32).astype(np.uint32)


def _philox4x32(counter: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], key: tuple[np.uint32, np.uint32]) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    c0, c1, c2, c3 = counter
    k0, k1 = key
    for _ in range(10):
        hi0, lo0 = _mulhilo(c0, _PHILOX_M0)
        hi1, lo1 = _mulhilo(c2, _PHILOX_M1)
        c0, c1, c2, c3 = hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0
        k0 = np.uint32(np.uint64(k0) + np.uint64(_PHILOX_W0))
        k1 = np.uint32(np.uint64(k1) + np.uint64(_PHILOX_W1))
    return c0, c1, c2, c3


def _key(seed: int, zoom: int) -> tuple[np.uint32, np.uint32]:
    digest = hashlib.sha256(b"wfcai.inference.noise.v1" + seed.to_bytes(8, "little") + zoom.to_bytes(2, "little")).digest()
    return np.uint32(int.from_bytes(digest[:4], "little")), np.uint32(int.from_bytes(digest[4:8], "little"))


def normal_field(*, seed: int, zoom: int, x: int, y: int, width: int, height: int) -> torch.Tensor:
    """Return a CPU float32 normal field addressed by global pixel coordinates."""
    columns = np.arange(x, x + width, dtype=np.int64)[None, :]
    rows = np.arange(y, y + height, dtype=np.int64)[:, None]
    x_words = columns.astype(np.uint64)
    y_words = rows.astype(np.uint64)
    c0 = np.broadcast_to((x_words & _MASK32).astype(np.uint32), (height, width))
    c1 = np.broadcast_to((x_words >> np.uint64(32)).astype(np.uint32), (height, width))
    c2 = np.broadcast_to((y_words & _MASK32).astype(np.uint32), (height, width))
    c3 = np.broadcast_to((y_words >> np.uint64(32)).astype(np.uint32), (height, width))
    word0, word1, _, _ = _philox4x32((c0, c1, c2, c3), _key(seed, zoom))
    u1 = (word0.astype(np.float64) + 0.5) / 4294967296.0
    u2 = (word1.astype(np.float64) + 0.5) / 4294967296.0
    values = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return torch.from_numpy(values.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)


def upscaler_noise(*, seed: int, zoom: int, x: int, y: int, width: int, height: int) -> torch.Tensor:
    """Return pixel-shuffle noise addressed at destination-pixel coordinates."""
    destination = normal_field(
        seed=seed,
        zoom=zoom,
        x=2 * x,
        y=2 * y,
        width=2 * width,
        height=2 * height,
    )
    return torch.cat([
        destination[..., dy::2, dx::2]
        for dy in range(2)
        for dx in range(2)
    ], dim=1)

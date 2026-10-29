"""A small private byte-bounded tensor LRU."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Hashable

import torch


@dataclass(frozen=True)
class CacheStats:
    bytes: int
    hits: int
    misses: int
    evictions: int


class TensorCache:
    def __init__(self, capacity_bytes: int) -> None:
        self._capacity_bytes = capacity_bytes
        self._entries: OrderedDict[Hashable, torch.Tensor] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable) -> torch.Tensor | None:
        try:
            value = self._entries.pop(key)
        except KeyError:
            self._misses += 1
            return None
        self._entries[key] = value
        self._hits += 1
        return value

    def put(self, key: Hashable, value: torch.Tensor) -> torch.Tensor:
        if self._capacity_bytes == 0:
            return value
        value = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        size = value.numel() * value.element_size()
        if size > self._capacity_bytes:
            return value
        # A contiguous batch slice can still retain the entire batch's storage.
        # Copy only those views so byte accounting measures the storage we retain.
        if value.untyped_storage().nbytes() != size or value.storage_offset() != 0:
            value = value.clone()
        old = self._entries.pop(key, None)
        if old is not None:
            self._bytes -= old.numel() * old.element_size()
        while self._entries and self._bytes + size > self._capacity_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= evicted.numel() * evicted.element_size()
            self._evictions += 1
        self._entries[key] = value
        self._bytes += size
        return value

    def clear(self) -> None:
        """Drop retained tiles; hit/miss/eviction counters are lifetime totals."""
        self._entries.clear()
        self._bytes = 0

    @property
    def stats(self) -> CacheStats:
        return CacheStats(self._bytes, self._hits, self._misses, self._evictions)

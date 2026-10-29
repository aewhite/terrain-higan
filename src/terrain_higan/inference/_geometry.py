"""Integer geometry for valid-convolution terrain requests."""

from collections.abc import Iterator

from .config import Region
from .constants import UPSCALER_SHRINK, Z0_SHRINK


def z0_input(region: Region) -> Region:
    halo = Z0_SHRINK // 2
    return Region(
        0,
        region.x - halo,
        region.y - halo,
        region.width + Z0_SHRINK,
        region.height + Z0_SHRINK,
    )


def parent_region(region: Region) -> Region:
    halo = UPSCALER_SHRINK // 2
    # Floor and ceil, rather than truncation, preserve registration across zero.
    left = (region.x - halo) // 2
    top = (region.y - halo) // 2
    right = -(-(region.x + region.width + halo) // 2)
    bottom = -(-(region.y + region.height + halo) // 2)
    return Region(region.zoom - 1, left, top, right - left, bottom - top)


def child_slices(region: Region, parent: Region) -> tuple[slice, slice]:
    halo = UPSCALER_SHRINK // 2
    left = region.x - (2 * parent.x + halo)
    top = region.y - (2 * parent.y + halo)
    return slice(top, top + region.height), slice(left, left + region.width)


def tile_coordinates(region: Region, size: int) -> Iterator[tuple[int, int]]:
    left, top = region.x // size, region.y // size
    right = (region.x + region.width - 1) // size
    bottom = (region.y + region.height - 1) // size
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            yield x, y


def validate_coordinate_range(region: Region) -> None:
    limit = 2 ** 62
    if not (-limit <= region.x < limit and -limit <= region.y < limit):
        raise ValueError("x and y must leave room for inference halos through z20")
    if region.x + region.width > limit or region.y + region.height > limit:
        raise ValueError("region extent exceeds the supported coordinate range")

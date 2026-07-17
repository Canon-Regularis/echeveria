"""Spatial contract types: a bounding box, one region, and an ordered set of regions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from phytovision.exceptions import ContractViolationError
from phytovision.types.arrays import Mask


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates (row/col, half-open on the max side)."""

    min_row: int
    min_col: int
    max_row: int
    max_col: int

    def __post_init__(self) -> None:
        if self.max_row <= self.min_row or self.max_col <= self.min_col:
            raise ContractViolationError(f"degenerate bbox: {self}")

    @property
    def height(self) -> int:
        return self.max_row - self.min_row

    @property
    def width(self) -> int:
        return self.max_col - self.min_col


@dataclass(frozen=True, slots=True)
class Region:
    """A single measurable region of an image (the whole plant, or later one leaf).

    Contract (upheld by every ``RegionProvider``): ``mask`` is a boolean array matching the source
    image with at least one ``True`` pixel, and ``label`` names the kind of region.
    """

    id: int
    label: str  # "plant" | "leaf" | ...
    mask: Mask
    bbox: BBox

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise ContractViolationError(f"Region.mask must be boolean, got {self.mask.dtype}")
        if self.mask.ndim != 2:
            raise ContractViolationError(f"Region.mask must be 2-D, got shape {self.mask.shape}")
        if not self.mask.any():
            raise ContractViolationError(f"Region {self.id!r} ({self.label}) has an empty mask")

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True, slots=True)
class RegionSet:
    """A non-empty, ordered collection of regions produced by a ``RegionProvider``.

    ``kind`` records what the regions represent so downstream stages behave identically whether a
    provider returned one whole-plant region or many leaf regions.
    """

    regions: tuple[Region, ...]
    kind: str  # "plant" | "leaf"
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.regions:
            raise ContractViolationError(
                "RegionSet must contain at least one region (LSP invariant)"
            )

    @property
    def is_per_leaf(self) -> bool:
        return self.kind == "leaf"

    def __iter__(self) -> Iterator[Region]:
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)

"""The ``RegionProvider`` contract, the pipeline's central extensibility seam.

Every provider maps ``(image, plant_mask)`` to a **non-empty** ``RegionSet``. Whether it returns one
whole-plant region or N leaf regions, downstream stages are identical, so any provider can replace
another. ``tests/test_region_providers.py`` checks this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from phytovision.types import BBox, Image, Mask, Region, RegionSet


class RegionProvider(ABC):
    @abstractmethod
    def regions(self, image: Image, plant_mask: Mask) -> RegionSet: ...


def bbox_of(mask: Mask) -> BBox:
    """Tight bounding box of a non-empty boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    (row_idx,) = np.where(rows)
    (col_idx,) = np.where(cols)
    return BBox(
        min_row=int(row_idx[0]),
        min_col=int(col_idx[0]),
        max_row=int(row_idx[-1]) + 1,
        max_col=int(col_idx[-1]) + 1,
    )


def region_from_mask(region_id: int, label: str, mask: Mask) -> Region:
    """Build a validated ``Region`` from a boolean mask (computes its bbox)."""
    return Region(id=region_id, label=label, mask=mask, bbox=bbox_of(mask))

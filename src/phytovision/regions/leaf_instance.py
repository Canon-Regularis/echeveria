"""Future region provider: one region per leaf, via a ``LeafInstanceSegmenter``.

This is a fully-wired drop-in for ``WholePlantRegionProvider``. It is intentionally functional today
except for the trained model behind it: inject any ``LeafInstanceSegmenter`` and every downstream
stage (phenotyping, aggregation, stress model, explainability) works unchanged, now per leaf.
"""

from __future__ import annotations

from phytovision.exceptions import SegmentationError
from phytovision.regions.base import RegionProvider, region_from_mask
from phytovision.segmentation.leaves.instance import (
    LeafInstanceSegmenter,
    NotYetTrainedLeafSegmenter,
)
from phytovision.types import Image, Mask, RegionSet


class LeafInstanceRegionProvider(RegionProvider):
    def __init__(self, segmenter: LeafInstanceSegmenter | None = None) -> None:
        self.segmenter = segmenter or NotYetTrainedLeafSegmenter()

    def regions(self, image: Image, plant_mask: Mask) -> RegionSet:
        leaf_masks = [m for m in self.segmenter.segment_leaves(image, plant_mask) if m.any()]
        if not leaf_masks:
            raise SegmentationError("leaf segmenter returned no non-empty leaf masks")
        regions = tuple(
            region_from_mask(region_id=i, label="leaf", mask=m) for i, m in enumerate(leaf_masks)
        )
        return RegionSet(regions=regions, kind="leaf", image_shape=plant_mask.shape)

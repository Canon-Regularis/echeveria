"""v1 region provider: the whole plant as a single region."""

from __future__ import annotations

from phytovision.exceptions import SegmentationError
from phytovision.regions.base import RegionProvider, region_from_mask
from phytovision.types import Image, Mask, RegionSet


class WholePlantRegionProvider(RegionProvider):
    """Yields exactly one region, the plant foreground. The v1 default.

    Phenotyping/aggregation then treat the whole plant as the unit of measurement. When the leaf
    module lands, swapping in ``LeafInstanceRegionProvider`` is the only change to get per-leaf
    measurements. This class and everything downstream stay put.
    """

    def regions(self, image: Image, plant_mask: Mask) -> RegionSet:
        if not plant_mask.any():
            raise SegmentationError(
                "plant mask is empty; the segmenter must return a non-empty foreground "
                "before regions can be provided"
            )
        region = region_from_mask(region_id=0, label="plant", mask=plant_mask)
        return RegionSet(regions=(region,), kind="plant", image_shape=plant_mask.shape)

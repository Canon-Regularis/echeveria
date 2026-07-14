"""Foreground segmentation by colourfulness in CIELab, for non-green succulents.

Excess Green misses red, purple, or blue plants. This thresholds Lab chroma (distance from grey),
so a strongly coloured plant stands out from a neutral background whatever its hue.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.color import rgb2lab
from skimage.filters import threshold_otsu

from phytovision.segmentation.base import PlantSegmenter
from phytovision.segmentation.cleanup import clean_mask
from phytovision.types import Image, Mask
from phytovision.validation import validate_rgb_image

logger = logging.getLogger(__name__)


class LabChromaSegmenter(PlantSegmenter):
    def __init__(
        self,
        min_object_fraction: float = 0.002,
        keep_largest: bool = False,
        closing_radius: int = 2,
    ) -> None:
        self.min_object_fraction = min_object_fraction
        self.keep_largest = keep_largest
        self.closing_radius = closing_radius

    def segment(self, image: Image) -> Mask:
        validate_rgb_image(image)
        lab = rgb2lab(image)
        chroma = np.hypot(lab[..., 1], lab[..., 2])  # sqrt(a^2 + b^2): distance from neutral grey
        mask = self._threshold(chroma)
        mask = clean_mask(
            mask,
            image.shape[:2],
            min_object_fraction=self.min_object_fraction,
            closing_radius=self.closing_radius,
            keep_largest=self.keep_largest,
        )
        if not mask.any():  # never hand downstream an empty frame
            logger.warning(
                "Lab chroma found no foreground; treating the entire frame as foreground"
            )
            mask = np.ones(image.shape[:2], dtype=bool)
        return mask

    @staticmethod
    def _threshold(chroma: np.ndarray) -> Mask:
        finite = chroma[np.isfinite(chroma)]
        if finite.size == 0 or np.allclose(finite, finite.flat[0]):
            return np.zeros(chroma.shape, dtype=bool)
        return chroma > threshold_otsu(finite)

"""Excess-Green plus Otsu foreground segmentation, the default that needs no training.

Excess Green (2g - r - b on chromatic coordinates) is a classic vegetation index. It is a light,
no-fit baseline. It is weak on red or blue succulents, so segmentation is a swappable stage.
See LabChromaSegmenter for a colour-agnostic alternative.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters import threshold_otsu

from phytovision.segmentation.base import PlantSegmenter
from phytovision.segmentation.cleanup import clean_mask
from phytovision.types import Image, Mask
from phytovision.validation import validate_rgb_image

logger = logging.getLogger(__name__)

_EPS = 1e-6


class ExGThresholdSegmenter(PlantSegmenter):
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
        mask = self._threshold(self._excess_green(image))
        mask = clean_mask(
            mask,
            image.shape[:2],
            min_object_fraction=self.min_object_fraction,
            closing_radius=self.closing_radius,
            keep_largest=self.keep_largest,
        )
        if not mask.any():  # never hand downstream an empty frame
            logger.warning("Excess-Green found no foreground; falling back to HSV saturation")
            mask = self._saturation_fallback(image)
        return mask

    @staticmethod
    def _excess_green(image: Image) -> np.ndarray:
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        total = r + g + b + _EPS
        rn, gn, bn = r / total, g / total, b / total
        return 2.0 * gn - rn - bn

    @staticmethod
    def _threshold(exg: np.ndarray) -> Mask:
        finite = exg[np.isfinite(exg)]
        if finite.size == 0 or np.allclose(finite, finite.flat[0]):
            return np.zeros(exg.shape, dtype=bool)
        return exg > threshold_otsu(finite)

    @staticmethod
    def _saturation_fallback(image: Image) -> Mask:
        """Fall back to HSV saturation when Excess-Green finds nothing."""
        from skimage.color import rgb2hsv

        sat = rgb2hsv(image)[..., 1]
        if np.allclose(sat, sat.flat[0]):
            logger.warning("saturation is uniform; treating the entire frame as foreground")
            return np.ones(image.shape[:2], dtype=bool)
        return sat > threshold_otsu(sat)

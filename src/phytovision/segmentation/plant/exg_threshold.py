"""Excess-Green + Otsu foreground segmentation — the v1 default (no training required).

Excess Green (ExG = 2g - r - b on chromatic coordinates) is a classic vegetation index. It is a
sensible, dependency-light baseline; it is *not* ideal for red/blue succulents, which is exactly why
segmentation is an injectable stage — swap this for a learned segmenter without touching anything.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    closing,
    disk,
    remove_small_holes,
    remove_small_objects,
)

from phytovision.segmentation.base import PlantSegmenter
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

        exg = self._excess_green(image)
        mask = self._threshold(exg)
        mask = self._clean(mask, image.shape[:2])

        if not mask.any():  # fallback keeps the contract: never hand downstream an empty frame
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
        thresh = threshold_otsu(finite)
        return exg > thresh

    def _clean(self, mask: Mask, shape: tuple[int, int]) -> Mask:
        min_size = max(1, int(self.min_object_fraction * shape[0] * shape[1]))
        # scikit-image >=0.26: `max_size` removes objects/holes up to that size (replaces min_size).
        mask = remove_small_objects(mask, max_size=min_size)
        mask = remove_small_holes(mask, max_size=min_size)
        if self.closing_radius > 0 and mask.any():
            mask = closing(mask, disk(self.closing_radius))
        if self.keep_largest and mask.any():
            mask = self._largest_component(mask)
        return mask

    @staticmethod
    def _largest_component(mask: Mask) -> Mask:
        labelled = label(mask)
        props = regionprops(labelled)
        if not props:
            return mask
        biggest = max(props, key=lambda p: p.area)
        return labelled == biggest.label

    @staticmethod
    def _saturation_fallback(image: Image) -> Mask:
        """When ExG finds nothing (e.g. a non-green succulent), fall back to HSV saturation."""
        from skimage.color import rgb2hsv

        sat = rgb2hsv(image)[..., 1]
        if np.allclose(sat, sat.flat[0]):
            logger.warning("saturation is uniform; treating the entire frame as foreground")
            return np.ones(image.shape[:2], dtype=bool)  # last resort: treat all as foreground
        return sat > threshold_otsu(sat)

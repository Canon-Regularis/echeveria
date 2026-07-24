"""Excess-Green plus Otsu foreground segmentation, the default that needs no training.

Excess Green (2g - r - b on chromatic coordinates) is a classic vegetation index. It is a light,
no-fit baseline. It is weak on red or blue succulents, so segmentation is a swappable stage.
See LabChromaSegmenter for a colour-agnostic alternative.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.filters import threshold_otsu

from phytovision._num import excess_green
from phytovision.segmentation.plant._threshold_base import ThresholdSegmenter
from phytovision.types import Image, Mask

logger = logging.getLogger(__name__)


class ExGThresholdSegmenter(ThresholdSegmenter):
    def _score_field(self, image: Image) -> np.ndarray:
        # Work in float: a uint8 image would overflow (r + g + b wraps past 255) before EPS lands.
        rgb = image.astype(np.float64)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        return excess_green(r, g, b)  # the package's one ExG definition, shared with the feature

    def _empty_fallback(self, image: Image) -> Mask:
        """Fall back to HSV saturation when Excess-Green finds nothing."""
        from skimage.color import rgb2hsv

        logger.warning("Excess-Green found no foreground; falling back to HSV saturation")
        sat = rgb2hsv(image)[..., 1]
        if np.allclose(sat, sat.flat[0]):
            logger.warning("saturation is uniform; treating the entire frame as foreground")
            return np.ones(image.shape[:2], dtype=bool)
        return sat > threshold_otsu(sat)

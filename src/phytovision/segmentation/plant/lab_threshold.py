"""Foreground segmentation by colourfulness in CIELab, for non-green succulents.

Excess Green misses red, purple, or blue plants. This thresholds Lab chroma (distance from grey),
so a strongly coloured plant stands out from a neutral background whatever its hue.
"""

from __future__ import annotations

import logging

import numpy as np
from skimage.color import rgb2lab

from phytovision.segmentation.plant._threshold_base import ThresholdSegmenter
from phytovision.types import Image, Mask

logger = logging.getLogger(__name__)


class LabChromaSegmenter(ThresholdSegmenter):
    def _score_field(self, image: Image) -> np.ndarray:
        lab = rgb2lab(image)
        return np.hypot(lab[..., 1], lab[..., 2])  # sqrt(a^2 + b^2): distance from neutral grey

    def _empty_fallback(self, image: Image) -> Mask:
        logger.warning("Lab chroma found no foreground; treating the entire frame as foreground")
        return np.ones(image.shape[:2], dtype=bool)

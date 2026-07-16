"""Shared skeleton for threshold-based plant segmenters.

Both the Excess-Green and the Lab-chroma segmenter reduce the image to one scalar field, split it
with Otsu, clean the resulting mask, and fall back when nothing clears the threshold. That shared
shape lives here as a template method; a subclass supplies only the scalar field and the fallback.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
from skimage.filters import threshold_otsu

from phytovision.segmentation.base import PlantSegmenter
from phytovision.segmentation.cleanup import clean_mask
from phytovision.types import Image, Mask
from phytovision.validation import validate_rgb_image


def threshold_field(field: np.ndarray) -> Mask:
    """Otsu-threshold a scalar field; an empty or uniform field yields an all-False mask."""
    finite = field[np.isfinite(field)]
    if finite.size == 0 or np.allclose(finite, finite.flat[0]):
        return np.zeros(field.shape, dtype=bool)
    return field > threshold_otsu(finite)


class ThresholdSegmenter(PlantSegmenter):
    """Segment by thresholding a scalar field with Otsu, then cleaning the mask.

    Subclasses implement ``_score_field`` (the scalar field whose high values mark plant pixels) and
    ``_empty_fallback`` (the mask to use when nothing clears the threshold).
    """

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
        mask = threshold_field(self._score_field(image))
        mask = clean_mask(
            mask,
            image.shape[:2],
            min_object_fraction=self.min_object_fraction,
            closing_radius=self.closing_radius,
            keep_largest=self.keep_largest,
        )
        if not mask.any():  # never hand downstream an empty frame
            mask = self._empty_fallback(image)
        return mask

    @abstractmethod
    def _score_field(self, image: Image) -> np.ndarray: ...

    @abstractmethod
    def _empty_fallback(self, image: Image) -> Mask: ...

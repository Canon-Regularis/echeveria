"""The ``Preprocessor`` contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phytovision.types import Image


class Preprocessor(ABC):
    """Normalizes a raw image into the canonical form the rest of the pipeline expects.

    Contract: returns an ``H x W x 3`` float32 RGB array with values in ``[0, 1]``. Downstream
    segmentation, feature extraction and models all assume this form, so any ``Preprocessor`` is
    substitutable for another.
    """

    @abstractmethod
    def process(self, image: Image) -> Image: ...

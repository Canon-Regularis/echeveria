"""The ``PlantSegmenter`` contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phytovision.types import Image, Mask


class PlantSegmenter(ABC):
    """Separates plant foreground from background.

    Contract: given an ``H x W x 3`` float image in ``[0, 1]``, return a boolean ``H x W`` mask
    (True = plant). Implementations must return a mask of the same height/width as the input.
    """

    @abstractmethod
    def segment(self, image: Image) -> Mask: ...

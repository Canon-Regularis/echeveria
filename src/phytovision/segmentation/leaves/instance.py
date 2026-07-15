"""Leaf instance segmentation: the interface every per-leaf segmenter implements.

``WatershedLeafSegmenter`` is a no-training implementation. A trained one would wrap a Mask R-CNN or
YOLO-seg model. Either drops in behind ``LeafInstanceRegionProvider`` with no downstream change (see
docs/OBJECTIVES.md for the leaf-module scope).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from phytovision.types import Image, Mask


class LeafInstanceSegmenter(ABC):
    """Produces one boolean mask per detected leaf, given the image and the plant foreground."""

    @abstractmethod
    def segment_leaves(self, image: Image, plant_mask: Mask) -> list[Mask]: ...


class NotYetTrainedLeafSegmenter(LeafInstanceSegmenter):
    """Placeholder so wiring/tests can reference the seam before a model exists."""

    def segment_leaves(self, image: Image, plant_mask: Mask) -> list[Mask]:
        raise NotImplementedError(
            "Leaf instance segmentation is descoped from v1 (see docs/OBJECTIVES.md). "
            "Provide a trained LeafInstanceSegmenter to enable per-leaf phenotyping."
        )

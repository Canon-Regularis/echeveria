"""Leaf instance segmentation — the reserved (future) interface.

Descoped from v1 (see docs/OBJECTIVES.md): there is no in-domain Echeveria leaf-instance set
yet. The interface is defined so the future module drops in behind ``LeafInstanceRegionProvider``
without any downstream change. A concrete implementation would typically wrap a trained Mask R-CNN /
YOLO-seg model.
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

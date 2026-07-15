"""Classical watershed leaf segmenter: a no-training baseline that splits the plant into leaves.

Distance-transform watershed over the plant foreground. Each local maximum of the distance
transform (a leaf lobe's centre) seeds a basin. It needs no trained model, so it unlocks per-leaf
traits (leaf_count, wilted_leaf_ratio) out of the box, at the cost of a rough split. A compact blob
with one distance peak (a single round rosette, say) stays one region rather than being over-split.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from phytovision.segmentation.leaves.instance import LeafInstanceSegmenter
from phytovision.types import Image, Mask


class WatershedLeafSegmenter(LeafInstanceSegmenter):
    def __init__(self, min_distance: int = 6, min_leaf_fraction: float = 0.02) -> None:
        self.min_distance = min_distance
        self.min_leaf_fraction = min_leaf_fraction

    def segment_leaves(self, image: Image, plant_mask: Mask) -> list[Mask]:
        if not plant_mask.any():
            return []
        distance = ndimage.distance_transform_edt(plant_mask)
        peaks = peak_local_max(
            distance, min_distance=self.min_distance, labels=plant_mask.astype(np.int32)
        )
        if len(peaks) <= 1:  # one lobe: do not over-split a compact plant
            return [plant_mask.copy()]

        markers = np.zeros(plant_mask.shape, dtype=np.int32)
        for index, (row, col) in enumerate(peaks, start=1):
            markers[row, col] = index
        labels = watershed(-distance, markers, mask=plant_mask)

        min_size = max(1, int(self.min_leaf_fraction * int(plant_mask.sum())))
        leaves = [labels == label for label in range(1, int(labels.max()) + 1)]
        leaves = [leaf for leaf in leaves if int(leaf.sum()) >= min_size]
        return leaves or [plant_mask.copy()]

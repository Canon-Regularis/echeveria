"""Skeleton and medial-axis descriptors of a region silhouette.

Honest scope: this skeletonizes the region MASK, the silhouette, not the vein network. It is a
morphological structure proxy read from the shape, not vein extraction, which would need
texture-level segmentation this does not do. Wilting thins and fragments the silhouette and shortens
the medial axis, so the skeleton length, its branching, and the medial thickness track structural
change. The descriptors are additive: this extractor is registered but not in the default stack, so
it never changes the existing feature schema unless a pipeline selects it.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import medial_axis

from phytovision._num import EPS
from phytovision.phenotyping.base import FeatureExtractor
from phytovision.types import Image, Region

_KEYS = ("length", "length_to_area", "branch_points", "endpoints", "mean_thickness", "tortuosity")
# 8-connectivity neighbour-counting kernel (the centre is excluded).
_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


class SkeletonFeatures(FeatureExtractor):
    namespace: ClassVar[str] = "skeleton"
    # Length and the branch and endpoint counts are extensive: they sum across leaves. Thickness,
    # tortuosity, and the length-to-area ratio are intensive and area-weighted-averaged instead.
    extensive: ClassVar[frozenset[str]] = frozenset({"length", "branch_points", "endpoints"})

    def _compute(self, image: Image, region: Region) -> dict[str, float]:
        mask = region.mask
        area = float(mask.sum())
        if min(mask.shape) < 3 or area < 3.0:  # too small to skeletonize meaningfully
            return dict.fromkeys(_KEYS, 0.0)

        skeleton, distance = medial_axis(mask, return_distance=True)
        length = float(skeleton.sum())
        if length == 0.0:
            return dict.fromkeys(_KEYS, 0.0)

        neighbours = convolve(skeleton.astype(np.uint8), _NEIGHBOURS, mode="constant", cval=0)
        endpoints = float(np.count_nonzero(skeleton & (neighbours == 1)))
        branch_points = float(np.count_nonzero(skeleton & (neighbours >= 3)))
        mean_thickness = float(distance[skeleton].mean())
        # Tortuosity: skeleton length against the bounding-box diagonal, a wrinkliness proxy that is
        # near one for a straight blade and grows as the silhouette meanders.
        diagonal = math.hypot(region.bbox.height, region.bbox.width)

        return {
            "length": length,
            "length_to_area": length / (area + EPS),
            "branch_points": branch_points,
            "endpoints": endpoints,
            "mean_thickness": mean_thickness,
            "tortuosity": length / (diagonal + EPS),
        }

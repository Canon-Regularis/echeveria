"""Boundary / morphology descriptors (roughness, concavity, radial variation).

Wilting changes leaf/rosette outline: curling and drooping increase boundary roughness and concavity
relative to a turgid plant.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.measure import find_contours, regionprops

from phytovision.phenotyping.base import FeatureExtractor, prop, single_region_props
from phytovision.types import Image, Region

_EPS = 1e-9


class MorphologyFeatures(FeatureExtractor):
    namespace = "morphology"

    def _compute(self, image: Image, region: Region) -> dict[str, float]:
        mask = region.mask
        props = single_region_props(region)

        area = float(props.area)
        perimeter = float(props.perimeter)
        convex_area = prop(props, "area_convex", "convex_area")
        convex_image = getattr(props, "image_convex", None)
        if convex_image is None:
            convex_image = props.convex_image  # older scikit-image
        convex_perimeter = float(regionprops(convex_image.astype(np.int32))[0].perimeter)

        radial_variation = _radial_variation(mask, props.centroid)
        # boundary_complexity = perimeter vs the perimeter of an equal-area circle (>= 1).
        equiv_circle_perimeter = 2.0 * math.sqrt(math.pi * area)

        return {
            "solidity": float(props.solidity),
            "concavity": (convex_area - area) / (convex_area + _EPS),
            "roughness": perimeter / (convex_perimeter + _EPS),
            "boundary_complexity": perimeter / (equiv_circle_perimeter + _EPS),
            "radial_variation": radial_variation,
        }


def _radial_variation(mask: np.ndarray, centroid: tuple[float, float]) -> float:
    """Coefficient of variation of centroid-to-boundary distance (a wrinkliness proxy)."""
    contours = find_contours(mask.astype(float), 0.5)
    if not contours:
        return 0.0
    boundary = max(contours, key=len)  # (row, col) points
    cy, cx = centroid
    d = np.hypot(boundary[:, 0] - cy, boundary[:, 1] - cx)
    mean = d.mean()
    return float(d.std() / (mean + _EPS)) if mean > 0 else 0.0

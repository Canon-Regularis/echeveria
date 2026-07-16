"""Geometric / shape descriptors of a region's mask."""

from __future__ import annotations

import math

import numpy as np

from phytovision._num import EPS
from phytovision.phenotyping.base import FeatureExtractor, prop, single_region_props
from phytovision.types import Image, Region


class GeometryFeatures(FeatureExtractor):
    namespace = "geometry"
    extensive = frozenset({"area_px", "convex_area"})  # summed across regions, not averaged

    def _compute(self, image: Image, region: Region) -> dict[str, float]:
        props = single_region_props(region)
        area = float(props.area)
        perimeter = float(props.perimeter)
        convex_area = prop(props, "area_convex", "convex_area")
        major = prop(props, "axis_major_length", "major_axis_length")
        minor = prop(props, "axis_minor_length", "minor_axis_length")
        image_area = float(region.mask.size)

        circularity = 4.0 * math.pi * area / (perimeter**2 + EPS)
        return {
            "area_px": area,
            "area_fraction": area / (image_area + EPS),
            "perimeter": perimeter,
            "convex_area": convex_area,
            "solidity": float(props.solidity),
            "extent": float(props.extent),
            "eccentricity": float(props.eccentricity),
            "equivalent_diameter": prop(props, "equivalent_diameter_area", "equivalent_diameter"),
            "major_axis_length": major,
            "minor_axis_length": minor,
            "aspect_ratio": major / (minor + EPS),
            "elongation": 1.0 - (minor / (major + EPS)),
            "circularity": float(np.clip(circularity, 0.0, 1.0)),
            "compactness": perimeter**2 / (area + EPS),
            "orientation": float(props.orientation),
        }

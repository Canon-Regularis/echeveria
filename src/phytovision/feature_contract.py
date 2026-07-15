"""Declared value ranges for plant features.

Finiteness is a hard contract, enforced at runtime by ``PlantFeatures``. Ranges are softer: only
features whose computation guarantees a range are declared here, and a contract test asserts the
pipeline stays within them. That catches an extractor drifting out of range without making the
runtime fragile on the many legitimately unbounded features (areas, entropies, axis lengths).

This module imports nothing from the package, so low-level types can depend on it without a cycle.
"""

from __future__ import annotations

from collections.abc import Mapping

# Feature key -> (low, high), inclusive. Only features with a mathematically guaranteed range.
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "colour.gcc_mean": (0.0, 1.0),
    "colour.greenness_ratio": (0.0, 1.0),
    "colour.yellow_fraction": (0.0, 1.0),
    "colour.brown_fraction": (0.0, 1.0),
    "colour.red_fraction": (0.0, 1.0),
    "colour.saturation_mean": (0.0, 1.0),
    "colour.value_mean": (0.0, 1.0),
    "geometry.solidity": (0.0, 1.0),
    "geometry.extent": (0.0, 1.0),
    "geometry.circularity": (0.0, 1.0),
    "geometry.area_fraction": (0.0, 1.0),
    "geometry.eccentricity": (0.0, 1.0),
    "morphology.solidity": (0.0, 1.0),
    "plant.canopy_coverage": (0.0, 1.0),
    "plant.wilted_leaf_ratio": (0.0, 1.0),
}


def range_violations(values: Mapping[str, float | None]) -> dict[str, float]:
    """Declared-bounded features that are out of range, mapping key to the offending value."""
    out: dict[str, float] = {}
    for key, (low, high) in FEATURE_BOUNDS.items():
        value = values.get(key)
        if value is not None and not low <= value <= high:
            out[key] = value
    return out

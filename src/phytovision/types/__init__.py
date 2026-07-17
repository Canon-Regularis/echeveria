"""Core, mostly-immutable data structures passed between pipeline stages.

These types are the contract every stage speaks. Keeping them small and validated is what lets
concrete stages be swapped freely: as long as a stage produces and consumes these shapes correctly,
the rest of the pipeline neither knows nor cares which implementation ran. They are grouped by role,
the array aliases, the spatial regions, the feature vectors, and the analysis results, and all are
re-exported here so ``from phytovision.types import Region`` and the like keep working.
"""

from __future__ import annotations

from phytovision.types.arrays import Image, Mask
from phytovision.types.features import FeatureVector, PlantFeatures
from phytovision.types.geometry import BBox, Region, RegionSet
from phytovision.types.results import (
    AnalysisReport,
    Explanation,
    Reason,
    StressAssessment,
)

__all__ = [
    "AnalysisReport",
    "BBox",
    "Explanation",
    "FeatureVector",
    "Image",
    "Mask",
    "PlantFeatures",
    "Reason",
    "Region",
    "RegionSet",
    "StressAssessment",
]

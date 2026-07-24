"""The ``FeatureAggregator`` contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from phytovision.types import FeatureVector, PlantFeatures, RegionSet


class FeatureAggregator(ABC):
    """Reduces per-region feature vectors to one plant-level vector.

    Contract: must not assume a particular region count. Instance-only outputs (for example
    ``leaf_count``) are ``None`` unless the regions are per-leaf, so one aggregator serves v1 (one
    region) and the future leaf module (N regions) unchanged.

    ``reduction_policy`` maps each feature key to one of ``"sum"`` (extensive: add across regions),
    ``"circular"`` (a wrapping quantity such as hue, averaged on the circle), ``"axial"`` (an
    undirected angle such as orientation, averaged on the doubled angle), or ``"mean"``, so the
    aggregator is driven by declared metadata rather than hardcoded key names. An implementation
    must treat an unrecognised kind as the area-weighted mean: averaging a circular or axial key
    linearly puts the result on the wrong side of its seam, not merely slightly off.
    """

    @abstractmethod
    def aggregate(
        self,
        regions: RegionSet,
        features: Sequence[FeatureVector],
        reduction_policy: Mapping[str, str] | None = None,
    ) -> PlantFeatures: ...

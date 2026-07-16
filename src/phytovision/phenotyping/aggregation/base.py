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

    ``reduction_policy`` maps each feature key to ``"sum"`` or ``"mean"``,
    so the aggregator is driven by declared metadata rather than hardcoded key names.
    """

    @abstractmethod
    def aggregate(
        self,
        regions: RegionSet,
        features: Sequence[FeatureVector],
        reduction_policy: Mapping[str, str] | None = None,
    ) -> PlantFeatures: ...

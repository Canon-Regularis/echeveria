"""Feature contract types: a per-region vector and the aggregated plant-level vector."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from phytovision.exceptions import ContractViolationError


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Features computed for a single region. Keys are namespaced (e.g. ``geometry.area``)."""

    region_id: int
    values: dict[str, float]

    def merged_with(self, other: FeatureVector) -> FeatureVector:
        """Combine two extractors' outputs for the same region; raises on key collision."""
        if other.region_id != self.region_id:
            raise ContractViolationError(
                f"cannot merge features across regions: {self.region_id} vs {other.region_id}"
            )
        clash = self.values.keys() & other.values.keys()
        if clash:
            raise ContractViolationError(
                f"feature key collision across extractors: {sorted(clash)}"
            )
        return FeatureVector(self.region_id, {**self.values, **other.values})


@dataclass(frozen=True, slots=True)
class PlantFeatures:
    """Plant-level feature vector after aggregation.

    ``values`` may contain ``None`` for instance-only traits (e.g. ``leaf_count``) when the regions
    were not per-leaf. Those slots populate automatically once a leaf provider is used.

    Every non-null value must be finite. Extractors coerce their own output, so this invariant only
    fires on a genuine bug in a construction path that skips that coercion.
    """

    values: dict[str, float | None]
    region_count: int
    per_region: tuple[FeatureVector, ...] = field(default_factory=tuple)

    @classmethod
    def from_values(cls, values: Mapping[str, float | None]) -> PlantFeatures:
        """Wrap a flat feature mapping as a single-region ``PlantFeatures``."""
        return cls(values=dict(values), region_count=1)

    def __post_init__(self) -> None:
        for key, value in self.values.items():
            if value is not None and not math.isfinite(value):
                raise ContractViolationError(f"feature {key!r} is not finite: {value}")

    def defined(self) -> dict[str, float]:
        """Only the non-null features, e.g. for feeding a model."""
        return {k: v for k, v in self.values.items() if v is not None}

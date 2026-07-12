"""Temporal leaf-death / senescence prediction — reserved (future) module.

The future module (docs/OBJECTIVES.md) consumes per-leaf ``PlantFeatures`` histories (produced
once a leaf ``RegionProvider`` and cross-time tracking exist) and predicts whether a given leaf
dies within a horizon. Defined here so the seam is explicit; no implementation ships in v1.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from phytovision.types import PlantFeatures


class LeafDeathPredictor(ABC):
    @abstractmethod
    def predict_leaf_death(
        self, feature_history: Sequence[PlantFeatures], horizons_days: Sequence[int]
    ) -> dict[int, float]:
        """Map each horizon (days) to P(leaf dies within that horizon)."""

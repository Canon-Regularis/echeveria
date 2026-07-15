"""Temporal leaf-death / decline forecasting.

A ``LeafDeathPredictor`` consumes a plant's ``PlantFeatures`` history and projects how stressed it
will be at future horizons. The shipped ``TrendLeafDeathPredictor`` is a linear trend extrapolation
of the stress score, not a fitted or validated prognostic, and it treats each observation as one
time step (so horizons are days under daily sampling). Per-leaf tracking is future work
(docs/OBJECTIVES).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

from phytovision.models.base import StressModel
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.temporal.forecast import project_scores
from phytovision.types import PlantFeatures


class LeafDeathPredictor(ABC):
    @abstractmethod
    def predict_leaf_death(
        self, feature_history: Sequence[PlantFeatures], horizons_days: Sequence[int]
    ) -> dict[int, float]:
        """Map each horizon (days) to a projected-stress risk in [0, 1]."""


class TrendLeafDeathPredictor(LeafDeathPredictor):
    """Score each observation with a stress model, then extrapolate the trajectory linearly."""

    name: ClassVar[str] = "trend-leaf-death-v0"

    def __init__(self, model: StressModel | None = None) -> None:
        self.model = model or HeuristicStressModel()

    def predict_leaf_death(
        self, feature_history: Sequence[PlantFeatures], horizons_days: Sequence[int]
    ) -> dict[int, float]:
        scores = [self.model.predict(features).score for features in feature_history]
        return project_scores(scores, horizons_days)

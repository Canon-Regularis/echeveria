"""The ``Explainer`` contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phytovision.models.base import StressModel
from phytovision.types import Explanation, PlantFeatures, StressAssessment


class Explainer(ABC):
    """Turns a model + its prediction into human-readable reasons.

    The model is typed as ``StressModel`` (the abstraction the pipeline holds). An explainer may
    require a narrower capability (e.g. ``ContributionModel``) and should degrade explicitly, not
    silently, when it is absent.
    """

    @abstractmethod
    def explain(
        self, model: StressModel, features: PlantFeatures, assessment: StressAssessment
    ) -> Explanation: ...

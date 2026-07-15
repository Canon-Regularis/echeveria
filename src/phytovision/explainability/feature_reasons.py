"""Feature-contribution explainer.

Works with any model that satisfies the ``ContributionModel`` protocol (the heuristic and the
gradient-boosted model both do). It depends only on that narrow interface, not on a concrete model
class, which is the point of interface segregation and dependency inversion.
"""

from __future__ import annotations

import logging

from phytovision.explainability._reasons import build_reasons
from phytovision.explainability.base import Explainer
from phytovision.models.base import ContributionModel, StressModel
from phytovision.types import Explanation, PlantFeatures, StressAssessment

logger = logging.getLogger(__name__)


class FeatureContributionExplainer(Explainer):
    def __init__(self, top_k: int = 6) -> None:
        self.top_k = top_k

    def explain(
        self, model: StressModel, features: PlantFeatures, assessment: StressAssessment
    ) -> Explanation:
        if not isinstance(model, ContributionModel):
            logger.warning(
                "model %r does not support feature contributions; returning no reasons",
                getattr(model, "name", type(model).__name__),
            )
            return Explanation(reasons=(), method="unavailable")

        reasons = build_reasons(model, features, model.contributions(features), self.top_k)
        return Explanation(reasons=reasons, method="feature-contribution")

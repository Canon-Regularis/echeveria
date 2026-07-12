"""Feature-contribution explainer.

Works with any model that satisfies the ``ContributionModel`` protocol (the heuristic and the
gradient-boosted model both do). It depends only on that narrow interface — not on a concrete model
class — which is the point of interface segregation + dependency inversion.
"""

from __future__ import annotations

import logging

from phytovision.explainability.base import Explainer
from phytovision.models.base import ContributionModel, StressModel
from phytovision.types import Explanation, PlantFeatures, Reason, StressAssessment

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

        contributions = model.contributions(features)
        reasons = []
        for key, contribution in contributions.items():
            if contribution == 0.0:
                continue
            value = features.values.get(key)
            increases = contribution > 0.0
            label = model.feature_label(key)
            reasons.append(
                Reason(
                    feature=key,
                    direction="increases" if increases else "decreases",
                    contribution=contribution,
                    value=float(value) if value is not None else float("nan"),
                    description=f"{label} {'raises' if increases else 'lowers'} the estimate",
                )
            )

        reasons.sort(key=lambda r: abs(r.contribution), reverse=True)
        return Explanation(reasons=tuple(reasons[: self.top_k]), method="feature-contribution")

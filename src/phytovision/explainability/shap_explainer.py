"""SHAP-based explainer (needs the ``ml`` extra).

Uses a model's ``shap_attribution`` (TreeSHAP for the gradient-boosted model) to rank features by
their SHAP value and to report a completeness (additivity) check. A model without SHAP support
degrades to no reasons, like the feature-contribution explainer when a model is not attributable.
"""

from __future__ import annotations

import logging

from phytovision.explainability._reasons import build_reasons
from phytovision.explainability.base import Explainer
from phytovision.models.base import ContributionModel, ShapExplainable, StressModel
from phytovision.types import Explanation, PlantFeatures, StressAssessment

logger = logging.getLogger(__name__)


class ShapExplainer(Explainer):
    def __init__(self, top_k: int = 6) -> None:
        self.top_k = top_k

    def explain(
        self, model: StressModel, features: PlantFeatures, assessment: StressAssessment
    ) -> Explanation:
        if not (isinstance(model, ShapExplainable) and isinstance(model, ContributionModel)):
            logger.warning(
                "model %r does not support SHAP; returning no reasons",
                getattr(model, "name", type(model).__name__),
            )
            return Explanation(reasons=(), method="shap-unavailable")
        try:
            result = model.shap_attribution(features)
        except ImportError as exc:
            logger.warning("SHAP unavailable: %s", exc)
            return Explanation(reasons=(), method="shap-unavailable")

        reasons = build_reasons(model, features, result.values, self.top_k)
        additivity_error = abs(
            result.base_value + sum(result.values.values()) - result.model_output
        )
        return Explanation(reasons=reasons, method="shap", additivity_error=additivity_error)

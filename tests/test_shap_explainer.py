"""SHAP explainer (Q8) and its additivity / ranking checks (Q9). Needs the ml extra plus shap."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("shap")

from phytovision.explainability.shap_explainer import ShapExplainer
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.registries import EXPLAINERS
from phytovision.types import PlantFeatures

_KEYS = ["colour.gcc_mean", "colour.yellow_fraction", "texture.entropy"]


def _training_data(per_class: int = 60):
    rng = np.random.default_rng(0)
    dicts, labels = [], []
    for _ in range(per_class):
        dicts.append(
            {
                _KEYS[0]: float(rng.normal(0.40, 0.02)),
                _KEYS[1]: float(abs(rng.normal(0.03, 0.02))),
                _KEYS[2]: float(rng.normal(2.5, 0.3)),
            }
        )
        labels.append(0)
    for _ in range(per_class):
        dicts.append(
            {
                _KEYS[0]: float(rng.normal(0.30, 0.02)),
                _KEYS[1]: float(abs(rng.normal(0.40, 0.05))),
                _KEYS[2]: float(rng.normal(4.5, 0.3)),
            }
        )
        labels.append(1)
    return dicts, labels


def _fitted() -> GradientBoostedStressModel:
    dicts, labels = _training_data()
    return GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)


_STRESSED = PlantFeatures(values={_KEYS[0]: 0.30, _KEYS[1]: 0.45, _KEYS[2]: 4.6}, region_count=1)


def test_shap_explainer_produces_reasons_and_additivity() -> None:
    model = _fitted()
    explanation = ShapExplainer().explain(model, _STRESSED, model.predict(_STRESSED))
    assert explanation.method == "shap"
    assert explanation.reasons
    assert explanation.additivity_error is not None
    assert explanation.additivity_error < 1e-4  # TreeSHAP completeness holds


def test_shap_explainer_degrades_without_support() -> None:
    heuristic = HeuristicStressModel()
    explanation = ShapExplainer().explain(heuristic, _STRESSED, heuristic.predict(_STRESSED))
    assert explanation.method == "shap-unavailable"
    assert explanation.reasons == ()
    assert explanation.additivity_error is None


def test_shap_is_registered_and_buildable() -> None:
    assert "shap" in EXPLAINERS.names()
    assert isinstance(EXPLAINERS.create("shap"), ShapExplainer)


def test_shap_reasons_skip_a_schema_drifted_feature() -> None:
    # A feature the model trained on but the live extractor did not produce (schema drift) is scored
    # by SHAP over the full vector, yet it must not surface as a reason with a NaN observed value:
    # build_reasons cites only measured features, so the digest never carries a NaN.
    model = _fitted()
    # _KEYS[2] is trained on but absent from the live features (extractor drift).
    drifted = PlantFeatures(values={_KEYS[0]: 0.30, _KEYS[1]: 0.45}, region_count=1)
    explanation = ShapExplainer().explain(model, drifted, model.predict(drifted))

    cited = {reason.feature for reason in explanation.reasons}
    assert _KEYS[2] not in cited  # the unmeasured feature is not cited
    assert all(not np.isnan(reason.value) for reason in explanation.reasons)


def test_shap_reasons_are_ranked_by_magnitude() -> None:
    # The reported reasons must be ordered by the strength of their SHAP attribution.
    model = _fitted()
    explanation = ShapExplainer().explain(model, _STRESSED, model.predict(_STRESSED))
    magnitudes = [abs(reason.contribution) for reason in explanation.reasons]
    assert magnitudes == sorted(magnitudes, reverse=True)

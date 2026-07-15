"""Counterfactual explanations (Q10): the smallest single-feature change that flips the verdict."""

from __future__ import annotations

from phytovision.explainability.counterfactual import counterfactuals
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.types import PlantFeatures


def test_counterfactual_flips_the_verdict() -> None:
    model = HeuristicStressModel()
    features = PlantFeatures(values={"colour.yellow_fraction": 0.5}, region_count=1)
    original = model.predict(features).label

    changes = counterfactuals(model, features)
    assert changes, "expected a single-feature change to flip the verdict"

    cf = changes[0]
    assert cf.feature == "colour.yellow_fraction"
    perturbed = PlantFeatures(values={cf.feature: cf.target_value}, region_count=1)
    assert model.predict(perturbed).label == cf.target_label
    assert model.predict(perturbed).label != original


def test_counterfactual_is_empty_without_a_bounded_feature() -> None:
    # texture.entropy is not declared-bounded, so there is nothing to search over.
    model = HeuristicStressModel()
    features = PlantFeatures(values={"texture.entropy": 3.0}, region_count=1)
    assert counterfactuals(model, features) == []

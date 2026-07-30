"""Counterfactual explanations (Q10): the smallest single-feature change that flips the verdict."""

from __future__ import annotations

import pytest

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


def test_counterfactual_returns_the_nearest_flip() -> None:
    # The search must return the SMALLEST flipping change, not any flip. For yellow_fraction 0.5
    # (stressed) the nearest flip lowers it to 0.2 (mild), a distance of 0.3. Mutating
    # `distance < best_distance` to `>` (or dropping the abs) returns the farthest flip instead; the
    # existing test does not catch that because it only asserts that some flip occurs.
    model = HeuristicStressModel()
    features = PlantFeatures(values={"colour.yellow_fraction": 0.5}, region_count=1)
    change = counterfactuals(model, features)[0]
    assert change.feature == "colour.yellow_fraction"
    assert change.target_value == pytest.approx(0.2)
    assert change.target_label == "mild"


def test_counterfactual_is_empty_without_a_bounded_feature() -> None:
    # texture.entropy is not declared-bounded, so there is nothing to search over.
    model = HeuristicStressModel()
    features = PlantFeatures(values={"texture.entropy": 3.0}, region_count=1)
    assert counterfactuals(model, features) == []

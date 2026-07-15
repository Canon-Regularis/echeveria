"""Every registered stress model upholds the StressModel contract (registry-driven).

Models that cannot be built by name (they need training data, like gradient-boosted) are skipped, so
the skip is visible in the test report rather than silently dropped.
"""

from __future__ import annotations

import pytest
from _invariants import assert_valid_assessment

from phytovision.exceptions import ConfigError
from phytovision.models.base import ContributionModel
from phytovision.registries import STRESS_MODELS

_NAMES = STRESS_MODELS.names()


def _build(name: str):
    try:
        return STRESS_MODELS.create(name)
    except (TypeError, ConfigError) as exc:
        pytest.skip(f"{name} cannot be built by name: {exc}")


def test_expected_models_are_registered() -> None:
    # Registry-driven parametrize auto-enrolls new models; this guards against losing a built-in.
    assert {"heuristic", "ensemble", "gradient-boosted"} <= set(_NAMES)


@pytest.mark.parametrize("name", _NAMES)
def test_model_predicts_a_valid_assessment(name, plant_features) -> None:
    model = _build(name)
    assert_valid_assessment(model.predict(plant_features))


@pytest.mark.parametrize("name", _NAMES)
def test_model_contributions_reference_known_features(name, plant_features) -> None:
    model = _build(name)
    if not isinstance(model, ContributionModel):
        pytest.skip(f"{name} is not a ContributionModel")
    contributions = model.contributions(plant_features)
    assert set(contributions).issubset(set(plant_features.values))

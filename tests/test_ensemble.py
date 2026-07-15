"""The soft-voting ensemble is a substitutable StressModel that stays explainable (F11)."""

from __future__ import annotations

import pytest

from phytovision.exceptions import ConfigError
from phytovision.models.base import ContributionModel, StressModel
from phytovision.models.stress.ensemble import EnsembleStressModel
from phytovision.registries import STRESS_MODELS
from phytovision.types import PlantFeatures, StressAssessment

_FEATURES = PlantFeatures(values={"colour.gcc_mean": 0.4}, region_count=1)


class _ConstantModel(StressModel):
    """A stress model with a fixed score, for deterministic ensemble arithmetic."""

    name = "constant"

    def __init__(self, score: float, confidence: float = 0.5) -> None:
        self._score = score
        self._confidence = confidence

    def predict(self, features: PlantFeatures) -> StressAssessment:
        return StressAssessment(self._score, self._confidence, "bucket", self.name)


class _ContribModel(_ConstantModel):
    def __init__(self, score: float, contribs: dict[str, float]) -> None:
        super().__init__(score)
        self._contribs = contribs

    def contributions(self, features: PlantFeatures) -> dict[str, float]:
        return dict(self._contribs)

    def feature_label(self, key: str) -> str:
        return f"label:{key}"


def test_score_is_the_weighted_mean_of_members() -> None:
    ensemble = EnsembleStressModel([_ConstantModel(0.2), _ConstantModel(0.8)])
    assert ensemble.predict(_FEATURES).score == pytest.approx(0.5)

    weighted = EnsembleStressModel([_ConstantModel(0.2), _ConstantModel(0.8)], weights=[3.0, 1.0])
    assert weighted.predict(_FEATURES).score == pytest.approx(0.35)


def test_score_lies_between_the_members() -> None:
    ensemble = EnsembleStressModel([_ConstantModel(0.1), _ConstantModel(0.9)])
    score = ensemble.predict(_FEATURES).score
    assert 0.1 < score < 0.9  # never worse than the worst member


def test_contributions_average_across_contributing_members() -> None:
    a = _ContribModel(0.6, {"colour.gcc_mean": 0.4, "texture.entropy": 0.2})
    b = _ContribModel(0.4, {"colour.gcc_mean": 0.0, "colour.yellow_fraction": 0.6})
    ensemble = EnsembleStressModel([a, b])
    contributions = ensemble.contributions(_FEATURES)
    assert contributions["colour.gcc_mean"] == pytest.approx(0.2)  # (0.4 + 0.0) / 2
    assert contributions["texture.entropy"] == pytest.approx(0.1)  # 0.2 present in one member only
    assert contributions["colour.yellow_fraction"] == pytest.approx(0.3)


def test_contributions_ignore_non_contributing_members() -> None:
    contributor = _ContribModel(0.6, {"colour.gcc_mean": 0.4})
    plain = _ConstantModel(0.4)
    ensemble = EnsembleStressModel([contributor, plain], weights=[1.0, 1.0])
    # The plain member has no contributions, so the contributor's weight renormalizes to 1.
    assert ensemble.contributions(_FEATURES) == {"colour.gcc_mean": pytest.approx(0.4)}


def test_contributions_empty_when_no_member_can_attribute() -> None:
    ensemble = EnsembleStressModel([_ConstantModel(0.3), _ConstantModel(0.7)])
    assert ensemble.contributions(_FEATURES) == {}


def test_feature_label_comes_from_a_contributing_member() -> None:
    ensemble = EnsembleStressModel([_ConstantModel(0.3), _ContribModel(0.7, {"k": 1.0})])
    assert ensemble.feature_label("k") == "label:k"
    # With no contributor knowing the key, the raw key is returned unchanged.
    assert EnsembleStressModel([_ConstantModel(0.3)]).feature_label("k") == "k"


def test_ensemble_is_a_stress_and_contribution_model() -> None:
    ensemble = EnsembleStressModel([_ContribModel(0.5, {"k": 0.1})])
    assert isinstance(ensemble, StressModel)
    assert isinstance(ensemble, ContributionModel)


@pytest.mark.parametrize(
    ("members", "weights", "match"),
    [
        ([], None, "at least one member"),
        ([_ConstantModel(0.5)], [1.0, 2.0], "does not match"),
        ([_ConstantModel(0.5)], [-1.0], "non-negative"),
        ([_ConstantModel(0.5)], [0.0], "positive value"),
    ],
)
def test_invalid_construction_raises(members, weights, match) -> None:
    with pytest.raises(ConfigError, match=match):
        EnsembleStressModel(members, weights=weights)


def test_registry_builds_ensemble_by_name() -> None:
    assert "ensemble" in STRESS_MODELS.names()
    model = STRESS_MODELS.create("ensemble", members=["heuristic", "heuristic"])
    assert isinstance(model, EnsembleStressModel)
    assert 0.0 <= model.predict(_FEATURES).score <= 1.0

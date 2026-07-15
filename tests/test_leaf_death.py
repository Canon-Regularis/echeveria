"""The reserved LeafDeathPredictor, implemented as a trend extrapolation over a feature history."""

from __future__ import annotations

from phytovision.models.temporal.leaf_death import LeafDeathPredictor, TrendLeafDeathPredictor
from phytovision.registries import LEAF_DEATH_PREDICTORS
from phytovision.types import PlantFeatures


def _features(gcc: float, yellow: float = 0.0) -> PlantFeatures:
    return PlantFeatures(
        values={"colour.gcc_mean": gcc, "colour.yellow_fraction": yellow}, region_count=1
    )


def test_predicts_rising_risk_over_horizons() -> None:
    # Greenness falling and yellowing rising is a worsening trajectory, so risk should not fall.
    history = [_features(0.42), _features(0.34, 0.20), _features(0.28, 0.40)]
    risk = TrendLeafDeathPredictor().predict_leaf_death(history, horizons_days=[1, 5])
    assert set(risk) == {1, 5}
    assert 0.0 <= risk[1] <= risk[5] <= 1.0


def test_registered_and_is_a_predictor() -> None:
    assert "trend" in LEAF_DEATH_PREDICTORS.names()
    predictor = LEAF_DEATH_PREDICTORS.create("trend")
    assert isinstance(predictor, LeafDeathPredictor)


def test_empty_and_single_history() -> None:
    predictor = TrendLeafDeathPredictor()
    assert predictor.predict_leaf_death([], [1, 3]) == {1: 0.0, 3: 0.0}
    single = predictor.predict_leaf_death([_features(0.3)], [1, 3])
    assert set(single) == {1, 3}

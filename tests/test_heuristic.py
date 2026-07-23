"""The heuristic stress model: its score, its bounds, and its numeric behaviour at the extremes."""

from __future__ import annotations

from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.types import PlantFeatures


def _features(**values: float) -> PlantFeatures:
    return PlantFeatures(values=dict(values), region_count=1)


def test_senescence_raises_the_score() -> None:
    model = HeuristicStressModel()
    healthy = model.predict(_features(**{"colour.yellow_fraction": 0.02}))
    wilting = model.predict(_features(**{"colour.yellow_fraction": 0.60}))
    assert wilting.score > healthy.score
    assert 0.0 <= healthy.score <= 1.0 and 0.0 <= wilting.score <= 1.0


def test_score_and_confidence_stay_in_range() -> None:
    model = HeuristicStressModel()
    values = {"colour.yellow_fraction": 0.3, "colour.gcc_mean": 0.35}
    assessment = model.predict(_features(**values))
    assert 0.0 <= assessment.score <= 1.0
    assert 0.0 <= assessment.confidence <= 1.0
    assert assessment.label in {"healthy", "mild", "stressed"}


def test_an_extreme_bias_saturates_instead_of_overflowing() -> None:
    # math.exp raises OverflowError past roughly 709, so an extreme bias used to crash the model
    # rather than settle at the end of the range. The logistic now saturates like the rest of the
    # numeric helpers clamp, so the score is a valid 0.0 or 1.0.
    features = _features(**{"colour.yellow_fraction": 0.1})
    assert HeuristicStressModel(bias=-800.0).predict(features).score == 0.0
    assert HeuristicStressModel(bias=800.0).predict(features).score == 1.0

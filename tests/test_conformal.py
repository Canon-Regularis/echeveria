"""Split conformal prediction sets (F13). Dependency-light: wraps any StressModel."""

from __future__ import annotations

import pytest

from phytovision.exceptions import ConfigError, ModelNotFittedError
from phytovision.models.base import StressModel
from phytovision.models.conformal import (
    ConformalSet,
    SplitConformalClassifier,
    conformal_quantile,
)
from phytovision.types import PlantFeatures, StressAssessment


class _ScoreModel(StressModel):
    """A model whose score is read straight from a feature, for controllable calibration."""

    name = "score"

    def predict(self, features: PlantFeatures) -> StressAssessment:
        score = min(1.0, max(0.0, float(features.values["s"])))
        return StressAssessment(score, 0.5, "x", self.name)


def _feat(score: float) -> PlantFeatures:
    return PlantFeatures(values={"s": score}, region_count=1)


def test_conformal_quantile_known_values() -> None:
    assert conformal_quantile([0.2, 0.4, 0.6, 0.8], 0.5) == pytest.approx(0.8)
    # Small n with tiny alpha clamps the level to 1, so the threshold is the max score.
    assert conformal_quantile([0.1, 0.9], 0.01) == pytest.approx(0.9)


def test_conformal_quantile_is_wider_for_smaller_alpha() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert conformal_quantile(scores, 0.05) >= conformal_quantile(scores, 0.5)


def test_conformal_quantile_rejects_empty() -> None:
    with pytest.raises(ConfigError, match="empty"):
        conformal_quantile([], 0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_alpha_must_be_in_unit_interval(alpha) -> None:
    with pytest.raises(ConfigError, match="alpha"):
        SplitConformalClassifier(_ScoreModel(), alpha=alpha)


def test_predict_set_before_calibrate_raises() -> None:
    clf = SplitConformalClassifier(_ScoreModel())
    with pytest.raises(ModelNotFittedError):
        clf.predict_set(_feat(0.5))


def test_calibrate_rejects_empty_and_mismatched() -> None:
    clf = SplitConformalClassifier(_ScoreModel())
    with pytest.raises(ConfigError, match="empty"):
        clf.calibrate([], [])
    with pytest.raises(ConfigError, match="same length"):
        clf.calibrate([_feat(0.5)], [0, 1])


def test_prediction_sets_are_calibrated() -> None:
    # Calibration: all healthy (label 0), so nonconformity == score. With these scores the threshold
    # qhat is 0.4, giving clean singleton and empty regions to check.
    features = [_feat(v) for v in (0.0, 0.1, 0.2, 0.3, 0.4)]
    labels = [0, 0, 0, 0, 0]
    clf = SplitConformalClassifier(_ScoreModel(), alpha=0.1).calibrate(features, labels)
    assert clf.qhat == pytest.approx(0.4)

    assert clf.predict_set(_feat(0.4)).labels == ("healthy",)  # score <= qhat only
    assert clf.predict_set(_feat(0.6)).labels == ("stressed",)  # 1 - score <= qhat only
    empty = clf.predict_set(_feat(0.5))  # neither label is within the threshold
    assert empty.labels == ()
    assert not empty.is_confident


def test_confident_set_reports_single_label() -> None:
    singleton = ConformalSet(("stressed",), score=0.9, alpha=0.1)
    both = ConformalSet(("healthy", "stressed"), score=0.5, alpha=0.1)
    assert singleton.is_confident
    assert not both.is_confident

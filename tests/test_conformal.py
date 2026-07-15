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
    # k = ceil((4 + 1) * 0.5) = 3, so the threshold is the 3rd smallest score.
    assert conformal_quantile([0.2, 0.4, 0.6, 0.8], 0.5) == pytest.approx(0.6)
    # k = ceil((2 + 1) * 0.99) = 3 > 2 scores, so the set is too small; the threshold is infinite.
    assert conformal_quantile([0.1, 0.9], 0.01) == float("inf")


def test_conformal_quantile_is_wider_for_smaller_alpha() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert conformal_quantile(scores, 0.05) >= conformal_quantile(scores, 0.5)


def test_conformal_quantile_rejects_empty() -> None:
    with pytest.raises(ConfigError, match="empty"):
        conformal_quantile([], 0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 1.5])
def test_conformal_quantile_rejects_bad_alpha(alpha) -> None:
    with pytest.raises(ConfigError, match="alpha"):
        conformal_quantile([0.1, 0.2], alpha)


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
    # Calibration: all healthy (label 0), so nonconformity == score. With alpha=0.5 and five scores,
    # k = ceil(6 * 0.5) = 3, so qhat is the 3rd smallest (0.3): clean singleton and empty regions.
    features = [_feat(v) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    labels = [0, 0, 0, 0, 0]
    clf = SplitConformalClassifier(_ScoreModel(), alpha=0.5).calibrate(features, labels)
    assert clf.qhat == pytest.approx(0.3)

    assert clf.predict_set(_feat(0.3)).labels == ("healthy",)  # score <= qhat only
    assert clf.predict_set(_feat(0.7)).labels == ("stressed",)  # 1 - score <= qhat only
    empty = clf.predict_set(_feat(0.5))  # neither label is within the threshold
    assert empty.labels == ()
    assert not empty.is_confident


def test_small_calibration_set_keeps_every_label() -> None:
    # k = ceil(6 * 0.9) = 6 > 5 scores, so the threshold is infinite and both labels are always kept
    # (coverage 1). This is the conservative small-sample behaviour, not under-coverage.
    features = [_feat(v) for v in (0.1, 0.2, 0.3, 0.4, 0.5)]
    clf = SplitConformalClassifier(_ScoreModel(), alpha=0.1).calibrate(features, [0] * 5)
    assert clf.qhat == float("inf")
    assert set(clf.predict_set(_feat(0.9)).labels) == {"healthy", "stressed"}
    assert set(clf.predict_set(_feat(0.1)).labels) == {"healthy", "stressed"}


def test_confident_set_reports_single_label() -> None:
    singleton = ConformalSet(("stressed",), score=0.9, alpha=0.1)
    both = ConformalSet(("healthy", "stressed"), score=0.5, alpha=0.1)
    assert singleton.is_confident
    assert not both.is_confident

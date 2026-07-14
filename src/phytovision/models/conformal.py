"""Split conformal prediction sets for water stress (F13).

A stress model returns one label. When the healthy/stressed boundary is genuinely ambiguous, that
hides the doubt. Split conformal wraps any ``StressModel`` and returns a *set* of labels with a
distribution-free coverage guarantee: over fresh data the true label lands in the set at least
``1 - alpha`` of the time. The set is one label when the model is confident, both labels when it is
not, and rarely empty when calibration says even the top label is unreliable.

This is composition, like the explainer: it never changes the wrapped model, only reads its scores.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from phytovision.exceptions import ConfigError, ModelNotFittedError
from phytovision.models.base import StressModel
from phytovision.types import PlantFeatures

_HEALTHY = "healthy"
_STRESSED = "stressed"


@dataclass(frozen=True, slots=True)
class ConformalSet:
    """A calibrated set of labels for one image, plus the score it was derived from."""

    labels: tuple[str, ...]  # subset of ("healthy", "stressed")
    score: float  # the wrapped model's P(stressed)
    alpha: float

    @property
    def is_confident(self) -> bool:
        """True when the set names exactly one label."""
        return len(self.labels) == 1


class SplitConformalClassifier:
    """Wrap a ``StressModel`` and turn its score into a calibrated label set."""

    def __init__(self, model: StressModel, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ConfigError(f"alpha must be in (0, 1), got {alpha}")
        self.model = model
        self.alpha = alpha
        self.qhat: float | None = None  # nonconformity threshold, set by calibrate()

    def calibrate(self, features: Sequence[PlantFeatures], labels: Sequence[int]) -> Self:
        """Fit the threshold on a held-out calibration split (1 = stressed, 0 = healthy)."""
        if len(features) != len(labels):
            raise ConfigError("features and labels must be the same length")
        if not features:
            raise ConfigError("calibration set is empty")
        scores = [
            self._nonconformity(self._score(feature), label)
            for feature, label in zip(features, labels, strict=True)
        ]
        self.qhat = conformal_quantile(scores, self.alpha)
        return self

    def predict_set(self, features: PlantFeatures) -> ConformalSet:
        """Return the calibrated label set for one image."""
        if self.qhat is None:
            raise ModelNotFittedError("call calibrate() before predict_set()")
        score = self._score(features)
        included: list[str] = []
        if (1.0 - score) <= self.qhat:  # nonconformity of "stressed" is within the threshold
            included.append(_STRESSED)
        if score <= self.qhat:  # nonconformity of "healthy" is within the threshold
            included.append(_HEALTHY)
        return ConformalSet(tuple(included), score, self.alpha)

    def _score(self, features: PlantFeatures) -> float:
        return self.model.predict(features).score

    @staticmethod
    def _nonconformity(score: float, label: int) -> float:
        """1 minus the model's probability of the true class."""
        prob_true = score if label == 1 else (1.0 - score)
        return 1.0 - prob_true


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The finite-sample-adjusted (1 - alpha) quantile of calibration nonconformity scores.

    Uses the conformal level ``ceil((n + 1)(1 - alpha)) / n``, clamped to 1, so a small calibration
    set stays conservative (a wider set) rather than under-covering.
    """
    n = len(scores)
    if n == 0:
        raise ConfigError("cannot take a conformal quantile of an empty set")
    level = min(1.0, math.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))

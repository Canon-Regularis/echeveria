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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

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

    # --- persistence ---
    def state(self) -> dict[str, object]:
        """The calibrated threshold, the miscoverage rate, and the wrapped model's own state."""
        from phytovision.models.persistence import Persistable

        if self.qhat is None:
            raise ModelNotFittedError("calibrate() before saving")
        if not isinstance(self.model, Persistable):
            raise ConfigError(f"wrapped model {type(self.model).__name__} cannot be saved")
        return {
            "alpha": self.alpha,
            "qhat": self.qhat,
            "model_type": self.model.MODEL_TYPE,
            "model_state": self.model.state(),
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> SplitConformalClassifier:
        from phytovision.models.persistence import model_from_state

        model = model_from_state(state["model_type"], state["model_state"])
        classifier = cls(model, alpha=state["alpha"])
        classifier.qhat = float(state["qhat"])
        return classifier

    def save(self, path: str | Path, manifest: Mapping[str, object] | None = None) -> None:
        """Persist the calibrated wrapper. Load only from trusted files (it unpickles)."""
        from phytovision.models.persistence import write_envelope

        write_envelope("conformal", self.state(), path, manifest)

    @classmethod
    def load(cls, path: str | Path) -> SplitConformalClassifier:
        from phytovision.models.persistence import read_envelope

        envelope = read_envelope(path)
        if envelope["model_type"] != "conformal":
            raise ConfigError(f"{path} is not a calibrated conformal model")
        return cls.from_state(envelope["state"])


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The split-conformal nonconformity threshold for a target coverage of ``1 - alpha``.

    The threshold is the k-th smallest calibration score, where ``k = ceil((n + 1)(1 - alpha))``.
    When ``k`` exceeds ``n`` the calibration set is too small to reach that level, so the threshold
    is infinite and every label is kept: conservative, never under-covering.
    """
    if not 0.0 < alpha < 1.0:
        raise ConfigError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    if n == 0:
        raise ConfigError("cannot take a conformal quantile of an empty set")
    # Subtract a tiny epsilon before the ceil so float error on an exact integer (n+1)(1-alpha)
    # landing at e.g. 941.0000000000001 cannot push k one rank too high and widen every set.
    k = math.ceil((n + 1) * (1.0 - alpha) - 1e-9)
    if k > n:
        return float("inf")
    return float(sorted(scores)[k - 1])

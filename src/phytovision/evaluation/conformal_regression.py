"""Split-conformal prediction intervals for a forecast, distribution-free.

The classifier in ``models.conformal`` turns a score into a calibrated label set; this reuses the
same primitive in residual space so any point forecaster gains a calibrated interval. Calibrate on a
held-out set of (prediction, actual) pairs: the interval half-width is the conformal quantile of the
absolute residuals, so over exchangeable data the true value lands in ``prediction +/- qhat`` at
least ``1 - alpha`` of the time. Time-series residuals are only approximately exchangeable, so treat
the coverage as close to nominal, not exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phytovision._num import clip01
from phytovision.exceptions import ConfigError
from phytovision.models.conformal import conformal_quantile


@dataclass(frozen=True, slots=True)
class ConformalIntervals:
    """A calibrated symmetric interval half-width for a point forecaster."""

    qhat: float
    alpha: float

    def interval(self, prediction: float, *, clip: bool = True) -> tuple[float, float]:
        """The calibrated interval around one point prediction, clipped to [0, 1] by default."""
        low = prediction - self.qhat
        high = prediction + self.qhat
        return (clip01(low), clip01(high)) if clip else (low, high)

    @classmethod
    def calibrate(
        cls, predictions: Sequence[float], actuals: Sequence[float], alpha: float = 0.1
    ) -> ConformalIntervals:
        """Fit the half-width from the absolute residuals of a held-out calibration set."""
        qhat = conformal_residual_quantile(predictions, actuals, alpha)
        return cls(qhat=qhat, alpha=alpha)


def conformal_residual_quantile(
    predictions: Sequence[float], actuals: Sequence[float], alpha: float = 0.1
) -> float:
    """The conformal quantile of the absolute residuals: the distribution-free half-width."""
    predicted = np.asarray(predictions, dtype=np.float64)
    actual = np.asarray(actuals, dtype=np.float64)
    if predicted.shape != actual.shape:
        raise ConfigError("predictions and actuals must be the same length")
    if predicted.size == 0:
        raise ConfigError("conformal calibration needs at least one residual")
    residuals = np.abs(actual - predicted)
    return conformal_quantile(residuals.tolist(), alpha)

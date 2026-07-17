"""Scoring rules for a predictive distribution, not just a point.

A forecaster that reports an interval should be judged on both calibration and sharpness, not on the
point error alone. These metrics do that: the continuous ranked probability score (CRPS) rewards a
mean that is close and an interval that is neither over-confident nor vacuous; the pinball loss
scores the quantiles the interval implies; the prediction-interval coverage (PICP) and mean width
report calibration against sharpness directly; the probability integral transform (PIT) turns a
calibrated forecast into a uniform sample. CRPS and PIT read the interval as a Gaussian, recovering
its standard deviation from the interval half-width, so they apply to any forecaster that reports a
central interval. All are means over the sample; lower is better for CRPS, pinball, and width.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from phytovision.exceptions import ContractViolationError

# The metrics accept either plain sequences or the numpy arrays returned by std_from_interval.
Floats = Sequence[float] | np.ndarray

# A floor on the recovered standard deviation, so a zero-width interval does not divide by zero.
_MIN_STD = 1e-6


@dataclass(frozen=True, slots=True)
class IntervalScore:
    """A predictive interval's calibration and sharpness against held-out truth."""

    crps: float
    pinball: float
    coverage: float
    mean_width: float
    n: int


def _as_arrays(*sequences: Floats) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(values, dtype=np.float64) for values in sequences)
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ContractViolationError("all inputs must be the same length")
    if arrays[0].size == 0:
        raise ContractViolationError("probabilistic scoring needs at least one observation")
    return arrays


def std_from_interval(lower: Floats, upper: Floats, level: float) -> np.ndarray:
    """Recover a Gaussian standard deviation from a central interval at ``level`` coverage."""
    low, high = _as_arrays(lower, upper)
    z = NormalDist().inv_cdf((1.0 + level) / 2.0)
    return np.maximum((high - low) / (2.0 * z), _MIN_STD)


def crps_gaussian_samples(actuals: Floats, means: Floats, stds: Floats) -> np.ndarray:
    """Per-observation CRPS reading each forecast as a Gaussian (Gneiting and Raftery form)."""
    from scipy.special import ndtr  # normal cdf; scipy is a core dependency

    actual, mean, std = _as_arrays(actuals, means, stds)
    std = np.maximum(std, _MIN_STD)
    z = (actual - mean) / std
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    return np.asarray(
        std * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi)), dtype=np.float64
    )


def crps_gaussian(actuals: Floats, means: Floats, stds: Floats) -> float:
    """Mean CRPS treating each forecast as a Gaussian; the closed form of Gneiting and Raftery."""
    return float(np.mean(crps_gaussian_samples(actuals, means, stds)))


def pinball_loss(actuals: Floats, quantiles: Floats, tau: float) -> float:
    """Mean pinball (quantile) loss for the ``tau`` quantile forecast ``quantiles``."""
    if not 0.0 < tau < 1.0:
        raise ContractViolationError(f"tau must be in (0, 1), got {tau}")
    actual, quantile = _as_arrays(actuals, quantiles)
    error = actual - quantile
    return float(np.mean(np.maximum(tau * error, (tau - 1.0) * error)))


def interval_pinball(
    actuals: Floats,
    means: Floats,
    lower: Floats,
    upper: Floats,
    level: float,
) -> float:
    """Average pinball loss across the interval's lower, median, and upper quantiles."""
    tau_low = (1.0 - level) / 2.0
    tau_high = (1.0 + level) / 2.0
    return float(
        np.mean(
            [
                pinball_loss(actuals, lower, tau_low),
                pinball_loss(actuals, means, 0.5),
                pinball_loss(actuals, upper, tau_high),
            ]
        )
    )


def interval_coverage(actuals: Floats, lower: Floats, upper: Floats) -> float:
    """The fraction of actuals inside the interval (PICP); compare it to the nominal level."""
    actual, low, high = _as_arrays(actuals, lower, upper)
    return float(np.mean((actual >= low) & (actual <= high)))


def mean_interval_width(lower: Floats, upper: Floats) -> float:
    """Mean interval width, a sharpness measure: narrower is sharper at equal coverage."""
    low, high = _as_arrays(lower, upper)
    return float(np.mean(high - low))


def pit_values(actuals: Floats, means: Floats, stds: Floats) -> np.ndarray:
    """The probability integral transform per observation; a calibrated forecast is uniform."""
    from scipy.special import ndtr

    actual, mean, std = _as_arrays(actuals, means, stds)
    std = np.maximum(std, _MIN_STD)
    return np.asarray(ndtr((actual - mean) / std), dtype=np.float64)


def interval_score(
    actuals: Floats,
    means: Floats,
    lower: Floats,
    upper: Floats,
    level: float,
) -> IntervalScore:
    """Bundle CRPS, pinball, coverage, and width for one set of interval forecasts."""
    stds = std_from_interval(lower, upper, level)
    return IntervalScore(
        crps=crps_gaussian(actuals, means, stds),
        pinball=interval_pinball(actuals, means, lower, upper, level),
        coverage=interval_coverage(actuals, lower, upper),
        mean_width=mean_interval_width(lower, upper),
        n=len(list(actuals)),
    )

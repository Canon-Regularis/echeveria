"""Forecast a plant's stress trajectory forward: a linear trend extrapolation, not a fitted model.

The high-throughput-phenotyping literature frames water stress as a trajectory. This fits a line to
the recent stress scores and projects it forward: an estimated steps-to-stressed, a projected score
per horizon, and a confidence that decays the further ahead it looks. It is an extrapolation of the
observed trend, not a validated prognostic, and each observation is treated as one time step (so
horizons are days under daily sampling).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from phytovision.temporal.history import Observation

_STRESSED_THRESHOLD = 0.66  # matches bucket_label's stressed cut-off
_RISING_TOLERANCE = 0.01  # slope per step at or below this is treated as flat
DEFAULT_HORIZONS = (1, 3, 7)


@dataclass(frozen=True, slots=True)
class Forecast:
    plant_id: str
    slope: float
    # The fitted trend's level at the last observation. The projection and steps_to_stressed both
    # anchor here (not the raw last reading), so a noisy final point cannot desync them.
    current_level: float
    projected_scores: dict[int, float]  # horizon step -> projected stress score in [0,1]
    steps_to_stressed: int | None  # steps until the projection crosses the stressed cut, else None
    confidence: float
    note: str


def stress_forecast(
    plant_id: str,
    series: Sequence[Observation],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> Forecast:
    """Project a plant's stress score forward by each horizon via a linear fit of its trajectory."""
    ordered = sorted(series, key=lambda obs: obs.timestamp)
    scores = [obs.stress_score for obs in ordered]
    steps = [h for h in horizons if h > 0]
    if len(scores) < 2:
        level = _clip01(scores[-1]) if scores else 0.0
        flat = dict.fromkeys(steps, level)
        return Forecast(
            plant_id, 0.0, level, flat, None, 0.1, "need two observations to project a trend"
        )

    slope, intercept, r2 = _fit_line(scores)
    end = len(scores) - 1
    current_level = intercept + slope * end  # the fitted line at the last step, not the raw reading
    projected = project_scores(scores, steps)
    steps_to_stressed = _steps_to_threshold(current_level, slope)
    confidence = _confidence(len(scores), r2, steps)
    return Forecast(
        plant_id,
        slope,
        _clip01(current_level),
        projected,
        steps_to_stressed,
        confidence,
        "linear trend extrapolation of the stress score",
    )


def project_scores(values: Sequence[float], horizons: Sequence[int]) -> dict[int, float]:
    """Project ``values`` forward by each horizon step via a linear fit, clipped to [0,1]."""
    if not values:
        return {h: 0.0 for h in horizons if h > 0}
    if len(values) < 2:
        return {h: _clip01(values[-1]) for h in horizons if h > 0}
    slope, intercept, _ = _fit_line(values)
    end = len(values) - 1
    return {h: _clip01(intercept + slope * (end + h)) for h in horizons if h > 0}


def _fit_line(values: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares slope, intercept, and R^2 of ``values`` against steps 0, 1, ... (len >= 2)."""
    n = len(values)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in enumerate(values))
    variance = sum((x - mean_x) ** 2 for x in range(n))
    slope = covariance / variance
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in values)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in enumerate(values))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, _clip01(r2)


def _steps_to_threshold(current_level: float, slope: float) -> int | None:
    """Steps until the fitted projection reaches the stressed cut, or None if not on the way there.

    Anchored on the fitted level at the last step (the same line ``project_scores`` uses), so the
    reported step always agrees with the projected score at that horizon.
    """
    if slope <= _RISING_TOLERANCE or current_level >= _STRESSED_THRESHOLD:
        return None
    return int(math.ceil((_STRESSED_THRESHOLD - current_level) / slope))


def _confidence(n: int, r2: float, horizons: Sequence[int]) -> float:
    """Confidence grows with trajectory length and fit quality, and decays with horizon distance."""
    data_factor = min(1.0, (n - 1) / 4.0)  # five or more observations reaches full weight
    max_horizon = max(horizons) if horizons else 1
    horizon_decay = 1.0 / (1.0 + max_horizon / n)
    return _clip01(r2 * data_factor * horizon_decay)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))

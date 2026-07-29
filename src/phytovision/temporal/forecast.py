"""Forecast a plant's stress trajectory forward, as a predictive distribution, not just a point.

The high-throughput-phenotyping literature frames water stress as a trajectory. The default fits a
line to the recent stress scores and projects it forward: an estimated steps-to-stressed, a
projected score per horizon, and now a prediction interval per horizon so the projection carries its
own uncertainty. Each observation is treated as one time step, so horizons are days under daily
sampling. This is an extrapolation of the observed trend, not a validated prognostic. Richer
forecasters (state-space, Gaussian process, Bayesian, ARIMA) live in ``models.forecasting`` and
return the same ``Forecast`` shape, so every surface reads them the same way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from phytovision._num import clip01
from phytovision.exceptions import ConfigError, ContractViolationError
from phytovision.models.base import STRESSED_THRESHOLD
from phytovision.temporal._fit import fit_line
from phytovision.temporal.history import Observation

DEFAULT_HORIZONS = (1, 3, 7)
DEFAULT_INTERVAL_LEVEL = 0.9

# A floor on the residual spread, so an exact two-point fit does not report a zero-width interval
# that would claim more certainty than two observations can support.
_MIN_RESIDUAL_STD = 0.02

# The horizon cap for the time-to-stressed search. It matches the forecasters' _MAX_LOOKAHEAD so the
# default linear forecast and the richer ones agree: beyond this, report no crossing rather than a
# meaningless far-future step from a near-flat slope.
_MAX_STEPS_TO_STRESSED = 30


def valid_interval_level(level: float) -> bool:
    """Whether a coverage level maps to a usable central-interval quantile.

    A level a hair below 1.0 (the largest double below one) passes ``0.0 < level < 1.0`` yet
    ``(1 + level) / 2`` rounds up to exactly 1.0 in float, which ``NormalDist().inv_cdf`` rejects.
    Guarding the mapped tail, not just the level, keeps the interval math from crashing.
    """
    return 0.0 < level < 1.0 and 0.0 < (1.0 + level) / 2.0 < 1.0


def t_multiplier(level: float, dof: int) -> float:
    """The two-sided Student-t multiplier for a central ``level`` interval with ``dof`` degrees.

    The residual spread is estimated from a handful of points, so a normal quantile makes the band
    too narrow and it undercovers at small n. The t multiplier widens it to the honest width and
    converges to the normal quantile as ``dof`` grows, so a long series is unaffected.
    """
    from scipy.stats import t as student_t

    return float(student_t.ppf((1.0 + level) / 2.0, max(1, dof)))


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
    # Per-horizon prediction interval at ``interval_level`` coverage. Empty for a degenerate series
    # with fewer than two observations. ``method`` names the forecaster that produced this.
    lower: dict[int, float] = field(default_factory=dict)
    upper: dict[int, float] = field(default_factory=dict)
    interval_level: float = DEFAULT_INTERVAL_LEVEL
    method: str = "linear-trend"
    # True when a statistical forecaster could not fit and silently substituted the linear interval,
    # so a consumer (the benchmark) can flag that its numbers are the fallback, not this method.
    degraded: bool = False


def stress_forecast(
    plant_id: str,
    series: Sequence[Observation],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> Forecast:
    """Project a plant's stress score forward by each horizon via a linear fit of its trajectory."""
    ordered = sorted(series, key=lambda obs: obs.timestamp)
    scores = [obs.stress_score for obs in ordered]
    return forecast_scores(plant_id, scores, horizons)


def forecast_scores(
    plant_id: str,
    scores: Sequence[float],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
) -> Forecast:
    """The linear-trend forecast over a raw score sequence, with a residual prediction interval."""
    if not valid_interval_level(interval_level):  # else it crashes inside NormalDist
        raise ConfigError(f"interval_level must be in (0, 1), got {interval_level}")
    if any(not math.isfinite(score) for score in scores):
        # A NaN score makes the line fit degenerate and silently projects a confident 0.0; the input
        # is invalid (the pipeline only ever produces finite, clipped scores), so reject it loudly.
        raise ContractViolationError("forecast series has a non-finite score (NaN or inf)")
    steps = [h for h in horizons if h > 0]
    if len(scores) < 2:
        level = clip01(scores[-1]) if scores else 0.0
        flat = dict.fromkeys(steps, level)
        return Forecast(
            plant_id,
            0.0,
            level,
            flat,
            None,
            0.1,
            "need two observations to project a trend",
            {},  # a degenerate series carries no interval, matching linear_prediction_interval
            {},
            interval_level,
            "linear-trend",
        )

    slope, intercept, r2 = fit_line(scores)
    end = len(scores) - 1
    current_level = intercept + slope * end  # the fitted line at the last step, not the raw reading
    projected = project_scores(scores, steps)
    lower, upper = linear_prediction_interval(scores, steps, interval_level)
    steps_to_stressed = _steps_to_threshold(intercept, slope, end)
    confidence = trend_confidence(len(scores), r2, steps)
    return Forecast(
        plant_id,
        slope,
        clip01(current_level),
        projected,
        steps_to_stressed,
        confidence,
        "linear trend extrapolation of the stress score",
        lower,
        upper,
        interval_level,
        "linear-trend",
    )


def project_scores(values: Sequence[float], horizons: Sequence[int]) -> dict[int, float]:
    """Project ``values`` forward by each horizon step via a linear fit, clipped to [0,1]."""
    if not values:
        return {h: 0.0 for h in horizons if h > 0}
    if len(values) < 2:
        return {h: clip01(values[-1]) for h in horizons if h > 0}
    slope, intercept, _ = fit_line(values)
    end = len(values) - 1
    return {h: clip01(intercept + slope * (end + h)) for h in horizons if h > 0}


def linear_prediction_interval(
    scores: Sequence[float], horizons: Sequence[int], level: float = DEFAULT_INTERVAL_LEVEL
) -> tuple[dict[int, float], dict[int, float]]:
    """Textbook ordinary-least-squares prediction interval per horizon, clipped to [0,1].

    The half-width is ``t * s * sqrt(1 + 1/n + (x0 - xbar)^2 / Sxx)``: it grows with the residual
    spread ``s`` and with the horizon's distance from the observed window, so a projection further
    ahead is honestly wider. The multiplier is a Student-t quantile on ``n - 2`` degrees of freedom,
    which is the exact OLS prediction interval and widens the band at the small samples here, where
    a normal quantile would undercover. A floor on ``s`` keeps a perfect fit from a zero-width band.
    """
    steps = [h for h in horizons if h > 0]
    n = len(scores)
    if n < 2 or not steps:
        return {}, {}
    slope, intercept, _ = fit_line(scores)
    mean_x = (n - 1) / 2.0
    sxx = sum((x - mean_x) ** 2 for x in range(n))
    residuals = [y - (intercept + slope * x) for x, y in enumerate(scores)]
    dof = max(1, n - 2)
    resid_std = max(_MIN_RESIDUAL_STD, (sum(r * r for r in residuals) / dof) ** 0.5)
    t = t_multiplier(level, dof)
    end = n - 1
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    for h in steps:
        x0 = end + h
        leverage = 1.0 + 1.0 / n + ((x0 - mean_x) ** 2 / sxx if sxx > 0 else 0.0)
        half = t * resid_std * leverage**0.5
        # Centre the band on the reported (clipped) projection. Centring on the raw mean lets a
        # projection past the ceiling clip both bounds to 1.0, collapsing the band to zero width and
        # feeding the probabilistic scorer a sigma of 0.
        mean = clip01(intercept + slope * x0)
        lower[h] = clip01(mean - half)
        upper[h] = clip01(mean + half)
    return lower, upper


def _steps_to_threshold(intercept: float, slope: float, end: int) -> int | None:
    """Steps until the fitted projection reaches the stressed cut, or None if it never does.

    Uses the exact ``intercept + slope * (end + step)`` that ``project_scores`` evaluates, and walks
    the step both ways until it is the true first crossing. So the reported step and the projected
    score can never disagree, even at a float boundary or a tiny slope. None only when the trend is
    flat or falling, or already at/above the cut.
    """
    # A non-finite score makes fit_line return a NaN slope, and every NaN comparison below is False,
    # so the guards would fall through to math.ceil(NaN) and raise. Report no crossing instead,
    # which is what the richer forecasters' _first_crossing already does on the same degenerate fit.
    if not math.isfinite(slope) or not math.isfinite(intercept):
        return None
    if slope <= 0.0 or intercept + slope * end >= STRESSED_THRESHOLD:
        return None
    step = max(1, int(math.ceil((STRESSED_THRESHOLD - (intercept + slope * end)) / slope)))
    if step > _MAX_STEPS_TO_STRESSED:
        return None  # beyond the search window: report no crossing, matching the richer forecasters
    while intercept + slope * (end + step) < STRESSED_THRESHOLD:  # walk off any float undershoot
        step += 1
    # And walk back off any overshoot: the division can round just above an exact integer crossing,
    # so ceil lands a step late and the forecast would report "2 steps away" while its own horizon-1
    # projection already sits at the cut.
    while step > 1 and intercept + slope * (end + step - 1) >= STRESSED_THRESHOLD:
        step -= 1
    return step if step <= _MAX_STEPS_TO_STRESSED else None


def trend_confidence(n: int, r2: float, horizons: Sequence[int]) -> float:
    """Confidence grows with trajectory length and fit quality, and decays with horizon distance."""
    data_factor = min(1.0, (n - 1) / 4.0)  # five or more observations reaches full weight
    max_horizon = max(horizons) if horizons else 1
    horizon_decay = 1.0 / (1.0 + max_horizon / n)
    return clip01(r2 * data_factor * horizon_decay)

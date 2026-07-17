"""Shared plumbing for the pluggable trajectory forecasters.

Each forecaster answers one question: given a plant's stress-score history, what will the score be
at each future horizon, and how sure are we. ``SeriesForecaster`` handles everything that is common
to all of them: the degenerate short-series case, the crossing-to-stressed search, the confidence
heuristic, and packing the result into a ``Forecast``. A concrete forecaster implements only
``_predict``, which returns a mean and an interval per future step. A numeric failure inside a
statistical backend falls back to the linear interval, so a single ill-conditioned series never
crashes a whole benchmark.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import ClassVar

from phytovision._num import clip01
from phytovision.exceptions import ConfigError
from phytovision.models.base import STRESSED_THRESHOLD, TrajectoryForecaster
from phytovision.temporal._fit import fit_line
from phytovision.temporal.forecast import (
    DEFAULT_HORIZONS,
    DEFAULT_INTERVAL_LEVEL,
    Forecast,
    linear_prediction_interval,
    project_scores,
    trend_confidence,
)

logger = logging.getLogger(__name__)

# How far ahead the crossing-to-stressed search looks when the requested horizons do not already
# reach the stressed cut. Beyond this the forecast reports no time-to-stressed rather than guess.
_MAX_LOOKAHEAD = 30


@dataclass(frozen=True, slots=True)
class Prediction:
    """A forecaster's raw output: a mean and an interval bound per future step."""

    mean: dict[int, float]
    lower: dict[int, float]
    upper: dict[int, float]


class SeriesForecaster(TrajectoryForecaster, ABC):
    """Template for a per-series forecaster. Subclasses implement ``_predict`` only."""

    name: ClassVar[str] = "series-forecaster"
    note: ClassVar[str] = "trajectory forecast"

    def __init__(self, interval_level: float = DEFAULT_INTERVAL_LEVEL) -> None:
        if not 0.0 < interval_level < 1.0:
            raise ConfigError(f"interval_level must be in (0, 1), got {interval_level}")
        self.interval_level = interval_level

    @abstractmethod
    def _predict(self, scores: Sequence[float], steps: Sequence[int]) -> Prediction:
        """Predict the mean and interval bounds at each future step (each >= 1)."""

    def forecast(
        self,
        series: Sequence[float],
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        plant_id: str = "",
    ) -> Forecast:
        """Fit this forecaster to ``series`` and project it forward to each horizon."""
        scores = list(series)
        steps = [h for h in horizons if h > 0]
        if len(scores) < 2:
            return self._flat(plant_id, scores, steps)

        lookahead = sorted(set(steps) | set(range(1, _MAX_LOOKAHEAD + 1)))
        try:
            prediction = self._predict(scores, lookahead)
        except ImportError:
            raise  # a missing extra is a clear, actionable error, not something to paper over
        except Exception as exc:  # numeric backends raise many types; degrade cleanly on any
            logger.warning("%s fell back to the linear interval: %s", self.name, exc)
            prediction = _linear_prediction(scores, lookahead, self.interval_level)

        projected = {h: prediction.mean[h] for h in steps}
        lower = {h: prediction.lower[h] for h in steps}
        upper = {h: prediction.upper[h] for h in steps}
        slope, intercept, r2 = fit_line(scores)
        current_level = clip01(intercept + slope * (len(scores) - 1))
        return Forecast(
            plant_id,
            slope,
            current_level,
            projected,
            _first_crossing(current_level, prediction.mean),
            trend_confidence(len(scores), r2, steps),
            self.note,
            lower,
            upper,
            self.interval_level,
            self.name,
        )

    def _flat(self, plant_id: str, scores: Sequence[float], steps: Sequence[int]) -> Forecast:
        """The degenerate forecast for a series too short to fit a trend."""
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
            dict(flat),
            dict(flat),
            self.interval_level,
            self.name,
        )


def _first_crossing(current_level: float, means: dict[int, float]) -> int | None:
    """The first future step whose mean reaches the stressed cut, or None.

    None when the plant is already at or above the cut (there is no time-to-stressed to report), or
    when no projected mean reaches it within the search window.
    """
    if current_level >= STRESSED_THRESHOLD:
        return None
    for step in sorted(means):
        if means[step] >= STRESSED_THRESHOLD:
            return step
    return None


def _linear_prediction(scores: Sequence[float], steps: Sequence[int], level: float) -> Prediction:
    """The linear-trend mean and interval, used as the common fallback."""
    mean = project_scores(scores, steps)
    lower, upper = linear_prediction_interval(scores, steps, level)
    return Prediction(mean, lower, upper)


def z_for(level: float) -> float:
    """The two-sided standard-normal multiplier for a central interval at ``level`` coverage."""
    return NormalDist().inv_cdf((1.0 + level) / 2.0)

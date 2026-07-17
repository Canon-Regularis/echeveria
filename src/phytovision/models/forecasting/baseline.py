"""The linear-trend forecaster, wrapped in the ``TrajectoryForecaster`` protocol.

This is the naive baseline every richer forecaster is measured against. It is the same linear
extrapolation the temporal package has always used, so its point projection matches the historical
``stress_forecast``; it now also carries a residual-based prediction interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from phytovision.exceptions import ConfigError
from phytovision.models.base import TrajectoryForecaster
from phytovision.temporal.forecast import (
    DEFAULT_HORIZONS,
    DEFAULT_INTERVAL_LEVEL,
    Forecast,
    forecast_scores,
)


class LinearTrendForecaster(TrajectoryForecaster):
    """Least-squares line fit projected forward, with an ordinary-least-squares interval."""

    name: ClassVar[str] = "linear-trend"

    def __init__(self, interval_level: float = DEFAULT_INTERVAL_LEVEL) -> None:
        if not 0.0 < interval_level < 1.0:  # match SeriesForecaster: reject at construction
            raise ConfigError(f"interval_level must be in (0, 1), got {interval_level}")
        self.interval_level = interval_level

    def forecast(
        self,
        series: Sequence[float],
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        plant_id: str = "",
    ) -> Forecast:
        return forecast_scores(plant_id, list(series), horizons, self.interval_level)

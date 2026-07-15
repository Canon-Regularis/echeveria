"""Temporal water-stress tracking: watch one plant over time, not just a single image.

The pieces are deliberately small. ``FeatureHistory`` stores per-plant observations keyed by a
sortable timestamp, and ``stress_trend`` reduces a plant's series to a direction and a slope. Both
build on the single-image pipeline, so nothing upstream changes.
"""

from __future__ import annotations

from phytovision.temporal.early_warning import EarlyWarning, pigment_early_warning
from phytovision.temporal.forecast import DEFAULT_HORIZONS, Forecast, stress_forecast
from phytovision.temporal.history import FeatureHistory, Observation
from phytovision.temporal.ingest import (
    build_history,
    plant_early_warnings,
    plant_forecasts,
    plant_trends,
)
from phytovision.temporal.trend import StressTrend, stress_trend

__all__ = [
    "DEFAULT_HORIZONS",
    "EarlyWarning",
    "FeatureHistory",
    "Forecast",
    "Observation",
    "StressTrend",
    "build_history",
    "pigment_early_warning",
    "plant_early_warnings",
    "plant_forecasts",
    "plant_trends",
    "stress_forecast",
    "stress_trend",
]

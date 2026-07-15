"""Temporal water-stress tracking: watch one plant over time, not just a single image.

The pieces are deliberately small. ``FeatureHistory`` stores per-plant observations keyed by a
sortable timestamp, and ``stress_trend`` reduces a plant's series to a direction and a slope. Both
build on the single-image pipeline, so nothing upstream changes.
"""

from __future__ import annotations

from phytovision.temporal.history import FeatureHistory, Observation
from phytovision.temporal.trend import StressTrend, stress_trend

__all__ = ["FeatureHistory", "Observation", "StressTrend", "stress_trend"]

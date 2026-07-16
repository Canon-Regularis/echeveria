"""Reduce a plant's stress-score history to a direction and a slope.

The slope is a least-squares fit of stress score against observation order (0, 1, 2, ...), not
against wall-clock time. Using order keeps irregular sampling from distorting the trend, and the
slope reads as "stress change per observation". A small dead band around zero reports "flat" so
sensor noise does not masquerade as a real trend.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from phytovision.temporal._fit import slope as fit_slope
from phytovision.temporal.history import Observation

# Slopes with magnitude below this (stress score per step) count as no meaningful change.
_FLAT_TOLERANCE = 0.01


@dataclass(frozen=True, slots=True)
class StressTrend:
    plant_id: str
    n: int
    slope: float  # change in stress score per observation step
    direction: str  # "rising" | "falling" | "flat" | "unknown"
    start_score: float
    end_score: float


def stress_trend(plant_id: str, series: Sequence[Observation]) -> StressTrend:
    """Fit a trend to a plant's observations. Sorts defensively, so any order is accepted."""
    ordered = sorted(series, key=lambda obs: obs.timestamp)
    scores = [obs.stress_score for obs in ordered]
    if not scores:
        return StressTrend(plant_id, 0, 0.0, "unknown", 0.0, 0.0)
    if len(scores) == 1:
        return StressTrend(plant_id, 1, 0.0, "flat", scores[0], scores[0])

    slope = fit_slope(scores)
    if slope > _FLAT_TOLERANCE:
        direction = "rising"
    elif slope < -_FLAT_TOLERANCE:
        direction = "falling"
    else:
        direction = "flat"
    return StressTrend(plant_id, len(scores), slope, direction, scores[0], scores[-1])

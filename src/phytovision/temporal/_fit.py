"""Least-squares line fitting over an evenly spaced series (steps 0, 1, 2, ...).

Kept separate so the trend, forecast, and early-warning modules share one fit, rather than each
carrying its own copy of the slope and intercept math.
"""

from __future__ import annotations

from collections.abc import Sequence

from phytovision._num import clip01


def fit_line(values: Sequence[float]) -> tuple[float, float, float]:
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
    return slope, intercept, clip01(r2)


def slope(values: Sequence[float]) -> float:
    """Least-squares slope of ``values`` against steps 0, 1, ... (len >= 2)."""
    return fit_line(values)[0]

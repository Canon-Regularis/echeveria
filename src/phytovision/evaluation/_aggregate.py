"""Shared aggregation for the evaluation harness.

The cross-validation, forecaster, and survival scorers all summarize per-fold numbers the same way:
a mean with a normal-approximation confidence interval. That one calculation lives here, so the
harnesses report their spread consistently instead of each carrying its own copy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mean_ci95(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """A 95% confidence interval for the mean via the normal approximation.

    Fewer than two values carry no spread, so the interval collapses to the mean.
    """
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array)) if array.size else 0.0
    if array.size < 2:
        return (mean, mean)
    half = 1.96 * float(np.std(array, ddof=1)) / (array.size**0.5)
    return (mean - half, mean + half)

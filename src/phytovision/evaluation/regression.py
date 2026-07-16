"""Regression metrics for the stress score against a continuous target, using numpy only.

The score is an RGB proxy, so these numbers show how well it tracks a measured water-status value.
They compare the score to the target directly, so give the target on a comparable scale, such as a
stress fraction in [0, 1]. RMSE and MAE are in the target's units; R2 is the share of the target's
variance the score accounts for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phytovision.exceptions import ContractViolationError


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Error of the score as a predictor of a continuous target."""

    rmse: float
    mae: float
    r2: float


def regression_metrics(scores: Sequence[float], targets: Sequence[float]) -> RegressionMetrics:
    """Root mean squared error, mean absolute error, and R2 of the score against the target."""
    score = np.asarray(scores, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if score.shape != target.shape:
        raise ContractViolationError("scores and targets must be the same length")
    if score.size == 0:
        raise ContractViolationError("regression needs at least one observation")

    error = score - target
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return RegressionMetrics(rmse=rmse, mae=mae, r2=r2)

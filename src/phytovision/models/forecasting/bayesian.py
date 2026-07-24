"""A Bayesian-ridge trend forecaster (scikit-learn, the ``ml`` extra).

Bayesian ridge regression fits the level and slope with a prior, so its forecast carries both the
parameter uncertainty and the noise variance. The predictive standard deviation widens the further
the horizon sits from the observed window, which is what an honest extrapolation should do.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from phytovision._num import clip01
from phytovision.models.forecasting.base import Prediction, SeriesForecaster, z_for
from phytovision.temporal.forecast import _MIN_RESIDUAL_STD


class BayesianRidgeForecaster(SeriesForecaster):
    name: ClassVar[str] = "bayesian-ridge"
    note: ClassVar[str] = "bayesian ridge trend with parameter and noise uncertainty"

    def _predict(self, scores: Sequence[float], steps: Sequence[int]) -> Prediction:
        try:
            from sklearn.linear_model import BayesianRidge
        except ImportError as exc:  # pragma: no cover - depends on the optional ml extra
            raise ImportError(
                'the bayesian-ridge forecaster needs the ml extra: pip install -e ".[ml]"'
            ) from exc

        n = len(scores)
        t = np.arange(n, dtype=float).reshape(-1, 1)
        model = BayesianRidge()
        model.fit(t, np.asarray(scores, dtype=float))

        end = n - 1
        future = np.array([[end + h] for h in steps], dtype=float)
        centre, std = model.predict(future, return_std=True)
        z = z_for(self.interval_level)

        mean: dict[int, float] = {}
        lower: dict[int, float] = {}
        upper: dict[int, float] = {}
        for h, point, spread in zip(steps, centre, std, strict=True):
            # Floor the spread like the other forecasters, so a smooth series of a few noisy points
            # cannot report a band tens of times too narrow (near-certain from a few readings).
            half = z * max(float(spread), _MIN_RESIDUAL_STD)
            # Centre the band on the clipped mean: a projection past the ceiling otherwise clips
            # both bounds to 1.0, so the recovered sigma degenerates to near-zero and overstates
            # confidence.
            projected = clip01(float(point))
            mean[h] = projected
            lower[h] = clip01(projected - half)
            upper[h] = clip01(projected + half)
        return Prediction(mean, lower, upper)

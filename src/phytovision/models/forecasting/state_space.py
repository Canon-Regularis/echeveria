"""A local-linear-trend state-space forecaster (statsmodels, the ``stats`` extra).

An unobserved-components model treats the score as a latent level and slope that evolve over time,
which is the classical way to track and forecast a drifting trajectory with proper uncertainty. It
gives a filtered state and a forecast whose prediction interval comes from the model itself, not
from a residual heuristic. A short or ill-conditioned series that the optimiser cannot fit falls
back to the linear interval upstream.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from phytovision._num import clip01
from phytovision.models.forecasting.base import Prediction, SeriesForecaster


def forecast_with_intervals(result: Any, steps: Sequence[int], level: float) -> Prediction:
    """Read a statsmodels forecast result into per-step mean and interval dicts, each in [0, 1]."""
    horizon = max(steps)
    forecast = result.get_forecast(steps=horizon)
    mean_all = np.asarray(forecast.predicted_mean, dtype=float)
    conf = np.asarray(forecast.conf_int(alpha=1.0 - level), dtype=float)
    mean: dict[int, float] = {}
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    for h in steps:
        index = h - 1  # statsmodels forecasts step 1 at position 0
        mean[h] = clip01(float(mean_all[index]))
        lower[h] = clip01(float(conf[index, 0]))
        upper[h] = clip01(float(conf[index, 1]))
    return Prediction(mean, lower, upper)


class StateSpaceForecaster(SeriesForecaster):
    name: ClassVar[str] = "state-space"
    note: ClassVar[str] = "local linear trend state-space forecast"

    def _predict(self, scores: Sequence[float], steps: Sequence[int]) -> Prediction:
        try:
            from statsmodels.tsa.statespace.structural import UnobservedComponents
        except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
            raise ImportError(
                'the state-space forecaster needs the stats extra: pip install -e ".[stats]"'
            ) from exc

        y = np.asarray(scores, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # statsmodels notes the missing date frequency; benign
            model = UnobservedComponents(y, level="local linear trend")
            result = model.fit(disp=False, maxiter=200)
        return forecast_with_intervals(result, steps, self.interval_level)

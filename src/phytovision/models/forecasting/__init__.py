"""Pluggable trajectory forecasters.

Every forecaster implements the ``TrajectoryForecaster`` protocol (a stress-score sequence in, a
``Forecast`` with prediction intervals out) and registers under ``FORECASTERS``. The linear baseline
and the two scikit-learn models need only the base or ``ml`` install; the state-space and ARIMA
models need the ``stats`` extra and import it lazily.
"""

from __future__ import annotations

from phytovision.models.forecasting.arima import ArimaForecaster
from phytovision.models.forecasting.base import Prediction, SeriesForecaster
from phytovision.models.forecasting.baseline import LinearTrendForecaster
from phytovision.models.forecasting.bayesian import BayesianRidgeForecaster
from phytovision.models.forecasting.gaussian_process import GaussianProcessForecaster
from phytovision.models.forecasting.state_space import StateSpaceForecaster

__all__ = [
    "ArimaForecaster",
    "BayesianRidgeForecaster",
    "GaussianProcessForecaster",
    "LinearTrendForecaster",
    "Prediction",
    "SeriesForecaster",
    "StateSpaceForecaster",
]

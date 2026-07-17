"""A Gaussian-process forecaster over time (scikit-learn, the ``ml`` extra).

The series is detrended with a least-squares line, the residual is modelled with a Gaussian process,
and the trend is added back. Detrending matters: a bare GP reverts to the series mean when it
extrapolates, which would ignore an ongoing decline; modelling only the residual lets the linear
trend carry the extrapolation while the GP supplies a predictive standard deviation that widens away
from the observed window.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from phytovision._num import clip01
from phytovision.models.forecasting.base import Prediction, SeriesForecaster, z_for
from phytovision.temporal._fit import fit_line


class GaussianProcessForecaster(SeriesForecaster):
    name: ClassVar[str] = "gaussian-process"
    note: ClassVar[str] = "gaussian process over the detrended score with a predictive interval"

    def _predict(self, scores: Sequence[float], steps: Sequence[int]) -> Prediction:
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        except ImportError as exc:  # pragma: no cover - depends on the optional ml extra
            raise ImportError(
                'the gaussian-process forecaster needs the ml extra: pip install -e ".[ml]"'
            ) from exc

        n = len(scores)
        slope, intercept, _ = fit_line(scores)
        t = np.arange(n, dtype=float)
        residual = np.asarray(scores, dtype=float) - (intercept + slope * t)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(2.0, (0.5, 20.0)) + WhiteKernel(
            0.01, (1e-6, 1.0)
        )
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False, random_state=0)
        with warnings.catch_warnings():
            # A near-perfect linear trend leaves almost no residual, so the optimiser can push a
            # kernel term to its bound. That is benign here: it means little residual structure.
            warnings.simplefilter("ignore")
            gp.fit(t.reshape(-1, 1), residual)

        end = n - 1
        future = np.array([[end + h] for h in steps], dtype=float)
        mean_residual, std = gp.predict(future, return_std=True)
        z = z_for(self.interval_level)

        mean: dict[int, float] = {}
        lower: dict[int, float] = {}
        upper: dict[int, float] = {}
        for h, residual_mean, spread in zip(steps, mean_residual, std, strict=True):
            centre = intercept + slope * (end + h) + float(residual_mean)
            half = z * float(spread)
            mean[h] = clip01(centre)
            lower[h] = clip01(centre - half)
            upper[h] = clip01(centre + half)
        return Prediction(mean, lower, upper)

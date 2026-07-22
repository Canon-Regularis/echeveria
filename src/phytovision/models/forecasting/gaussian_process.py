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
from phytovision.temporal.forecast import _MIN_RESIDUAL_STD


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
        mean_x = (n - 1) / 2.0
        sxx = float(sum((x - mean_x) ** 2 for x in range(n)))
        # The GP models the residual, but the detrending line is extrapolated, so its OLS parameter
        # uncertainty (growing with the horizon) must be added in quadrature, or the band ignores
        # the extrapolation risk and is overconfident far from the window. The residual std is
        # floored like the linear interval, so a near-perfect fit does not claim near-zero spread.
        resid_std = max(_MIN_RESIDUAL_STD, (float(np.sum(residual**2)) / max(1, n - 2)) ** 0.5)
        future = np.array([[end + h] for h in steps], dtype=float)
        mean_residual, std = gp.predict(future, return_std=True)
        z = z_for(self.interval_level)

        mean: dict[int, float] = {}
        lower: dict[int, float] = {}
        upper: dict[int, float] = {}
        for h, residual_mean, spread in zip(steps, mean_residual, std, strict=True):
            x0 = end + h
            leverage = 1.0 / n + ((x0 - mean_x) ** 2 / sxx if sxx > 0 else 0.0)
            total_std = (float(spread) ** 2 + resid_std**2 * leverage) ** 0.5
            centre = intercept + slope * x0 + float(residual_mean)
            half = z * total_std
            # Centre the band on the reported (clipped) mean. Centring on the raw projection lets a
            # forecast past the ceiling clip both bounds to 1.0, collapsing the band to zero width
            # and handing the probabilistic scorer a near-zero sigma it reads as near-certain.
            point = clip01(centre)
            mean[h] = point
            lower[h] = clip01(point - half)
            upper[h] = clip01(point + half)
        return Prediction(mean, lower, upper)

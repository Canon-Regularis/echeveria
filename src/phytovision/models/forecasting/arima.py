"""An ARIMA forecaster (statsmodels, the ``stats`` extra).

A differenced autoregressive-moving-average model is the standard time-series baseline above a plain
trend line. It supplies its own prediction interval, so the forecast reports uncertainty from the
fitted noise process rather than a residual heuristic. A series the optimiser cannot fit falls back
to the linear interval upstream.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import ClassVar

import numpy as np

from phytovision.models.forecasting.base import Prediction, SeriesForecaster
from phytovision.models.forecasting.state_space import forecast_with_intervals

# A modest order that fits short phenotyping series: one autoregressive term, one difference for the
# trend, one moving-average term.
_ORDER = (1, 1, 1)


class ArimaForecaster(SeriesForecaster):
    name: ClassVar[str] = "arima"
    note: ClassVar[str] = "arima forecast with native prediction intervals"

    def _predict(self, scores: Sequence[float], steps: Sequence[int]) -> Prediction:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
            raise ImportError(
                'the arima forecaster needs the stats extra: pip install -e ".[stats]"'
            ) from exc

        y = np.asarray(scores, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence and frequency notes on short series
            model = ARIMA(y, order=_ORDER)
            result = model.fit()
        return forecast_with_intervals(result, steps, self.interval_level)

"""Shared machinery for the covariate survival models (Weibull AFT and Cox).

Both fit a regression of time-to-wilt on the two observable covariates, then predict a per-plant
median with a central band. Everything they share lives here: z-scoring the covariates with the
cohort statistics stored at fit and reused at predict, dropping a constant column that would make
the information matrix singular, catching a non-convergent fit and falling back to the Kaplan-Meier
cohort baseline, and reading the per-plant median and its interquartile-time band. A subclass only
supplies its lifelines fitter.
"""

from __future__ import annotations

import logging
import warnings
from abc import abstractmethod
from typing import Any, ClassVar, Self

import numpy as np

from phytovision.models.survival.base import (
    _EXTRA_HINT,
    PlantSurvival,
    SurvivalDataset,
    SurvivalModel,
    _finite_or_none,
)
from phytovision.models.survival.kaplan_meier import KaplanMeierSurvival

logger = logging.getLogger(__name__)

# A covariate whose spread is below this is constant, which drives lifelines to a singular fit; drop
# it. A cohort with fewer events than this cannot support a covariate fit, so fall back to baseline.
_MIN_STD = 1e-9
_MIN_EVENTS = 2

# The band is the central 50 percent time-to-event interval. lifelines orients predict_percentile by
# the surviving fraction: the 0.75 quantile is the earlier time (lower), 0.25 the later (upper).
_LOWER_PERCENTILE = 0.75
_UPPER_PERCENTILE = 0.25


class CovariateSurvival(SurvivalModel):
    """A survival model over standardized covariates, with a Kaplan-Meier fallback."""

    uses_covariates: ClassVar[bool] = True
    penalizer: ClassVar[float] = 0.0

    def __init__(self, penalizer: float | None = None) -> None:
        self._penalizer = type(self).penalizer if penalizer is None else penalizer
        self._fitter: Any = None
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._kept: tuple[str, ...] = ()
        self._fallback: KaplanMeierSurvival | None = None

    @abstractmethod
    def _make_fitter(self) -> Any:
        """Return a lifelines fitter configured with this model's penalizer (lazily imported)."""

    def fit(self, dataset: SurvivalDataset) -> Self:
        names = dataset.covariate_names()
        frame = dataset.covariate_frame()
        self._means, self._stds = _column_stats(frame, names)
        self._kept = tuple(name for name in names if self._stds[name] >= _MIN_STD)
        if not self._kept or dataset.n_events < _MIN_EVENTS:
            return self._degrade(dataset, "too few events or no varying covariate")

        pandas = _pandas()
        fit_frame = _standardized_frame(dataset, self._kept, self._means, self._stds, pandas)
        fit_frame["_duration"] = dataset.durations()
        fit_frame["_event"] = dataset.events()
        fitter = self._make_fitter()  # a missing extra raises ImportError here, before the try
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # convergence notes on short synthetic cohorts
                self._fitter = fitter.fit(fit_frame, duration_col="_duration", event_col="_event")
        except Exception as exc:  # numeric non-convergence: degrade to the honest cohort baseline
            return self._degrade(dataset, f"fit did not converge ({type(exc).__name__})")
        return self

    def predict(self, dataset: SurvivalDataset) -> dict[str, PlantSurvival]:
        if self._fitter is None:
            return self._fallback_predict(dataset)
        pandas = _pandas()
        frame = _standardized_frame(dataset, self._kept, self._means, self._stds, pandas)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                median = self._fitter.predict_median(frame)
                lower = self._fitter.predict_percentile(frame, p=_LOWER_PERCENTILE)
                upper = self._fitter.predict_percentile(frame, p=_UPPER_PERCENTILE)
        except Exception as exc:  # a degenerate covariate profile at predict time
            logger.warning("%s prediction fell back to the cohort baseline: %s", self.name, exc)
            self._fallback = KaplanMeierSurvival().fit(dataset)
            self._fitter = None
            return self._fallback_predict(dataset)

        return {
            plant_id: PlantSurvival(
                plant_id,
                _finite_or_none(_at(median, index)),
                _finite_or_none(_at(lower, index)),
                _finite_or_none(_at(upper, index)),
                self.name,
                None,
            )
            for index, plant_id in enumerate(dataset.plant_ids())
        }

    def _degrade(self, dataset: SurvivalDataset, reason: str) -> Self:
        logger.warning("%s fell back to the Kaplan-Meier baseline: %s", self.name, reason)
        self._fallback = KaplanMeierSurvival().fit(dataset)
        self._fitter = None
        return self

    def _fallback_predict(self, dataset: SurvivalDataset) -> dict[str, PlantSurvival]:
        if self._fallback is None:
            self._fallback = KaplanMeierSurvival().fit(dataset)
        return self._fallback.predict(dataset)


def _pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
        raise ImportError(_EXTRA_HINT) from exc
    return pandas


def _column_stats(
    frame: list[dict[str, float]], names: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-covariate mean and true population standard deviation over the cohort.

    The std is unfloored, so ``fit`` can tell a constant column (std ~ 0) from a varying one and
    drop it. Flooring here would make every column look varying and defeat that drop; the
    divide-by-zero it guarded against cannot happen, since only kept columns are standardized.
    """
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in names:
        column = [row[name] for row in frame]
        mean = sum(column) / len(column)
        variance = sum((value - mean) ** 2 for value in column) / len(column)
        means[name] = mean
        stds[name] = variance**0.5
    return means, stds


def _standardized_frame(
    dataset: SurvivalDataset,
    kept: tuple[str, ...],
    means: dict[str, float],
    stds: dict[str, float],
    pandas: Any,
) -> Any:
    """A per-plant DataFrame of z-scored covariates, in dataset order."""
    rows = [
        {name: (record.covariates[name] - means[name]) / stds[name] for name in kept}
        for record in dataset.records
    ]
    return pandas.DataFrame(rows, columns=list(kept))


def _at(prediction: Any, index: int) -> Any:
    """Read the value at a positional index from a lifelines prediction.

    lifelines returns a pandas Series for a multi-row frame but squeezes a single-row prediction to
    a bare scalar, which has no ``.iloc``. Coercing to a 1-d array first handles both, so a one
    plant cohort (a leave-one-out survival fold) reads back cleanly instead of raising an error.
    """
    return np.atleast_1d(np.asarray(prediction, dtype=float))[index]

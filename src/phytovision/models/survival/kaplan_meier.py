"""The Kaplan-Meier cohort baseline: an honest survival curve with no covariates.

Kaplan-Meier is the non-parametric estimate of cohort survival under right censoring. It owns the
survival curve every fit displays, its cohort median, and a genuine 95 percent confidence band. It
carries no covariates, so it cannot rank plants: its per-plant estimate is the cohort median
broadcast to every plant, labelled ``cohort-km`` so it is never read as per-plant discrimination.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar, Self

from phytovision.models.survival.base import (
    _EXTRA_HINT,
    PlantSurvival,
    SurvivalCurve,
    SurvivalDataset,
    SurvivalModel,
    _finite_or_none,
)


class KaplanMeierSurvival(SurvivalModel):
    name: ClassVar[str] = "kaplan-meier"
    note: ClassVar[str] = "Kaplan-Meier cohort baseline, no covariates"
    uses_covariates: ClassVar[bool] = False

    def __init__(self) -> None:
        self._curve = SurvivalCurve((), ())
        self._median: float | None = None
        self._median_ci: tuple[float | None, float | None] = (None, None)

    def fit(self, dataset: SurvivalDataset) -> Self:
        fitter = _kaplan_meier_fitter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence and low-sample notes on synthetic series
            fitter.fit(dataset.durations(), dataset.events())
        self._curve = _curve_from(fitter)
        self._median = _finite_or_none(fitter.median_survival_time_)
        self._median_ci = _median_ci_from(fitter)
        return self

    def predict(self, dataset: SurvivalDataset) -> dict[str, PlantSurvival]:
        low, high = self._median_ci
        return {
            plant_id: PlantSurvival(plant_id, self._median, low, high, "cohort-km", self._curve)
            for plant_id in dataset.plant_ids()
        }

    def concordance_index(self, dataset: SurvivalDataset) -> float | None:
        return None  # a covariate-free model cannot rank plants

    def cohort_curve(self) -> SurvivalCurve:
        return self._curve

    def cohort_median(self) -> float | None:
        return self._median

    def cohort_median_ci(self) -> tuple[float | None, float | None]:
        return self._median_ci


def _kaplan_meier_fitter() -> Any:
    try:
        from lifelines import KaplanMeierFitter
    except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
        raise ImportError(_EXTRA_HINT) from exc
    return KaplanMeierFitter()


def _curve_from(fitter: Any) -> SurvivalCurve:
    survival = fitter.survival_function_
    interval = fitter.confidence_interval_
    times = tuple(float(t) for t in survival.index)
    return SurvivalCurve(
        times=times,
        survival=tuple(float(v) for v in survival.iloc[:, 0]),
        lower=tuple(float(v) for v in interval.iloc[:, 0]),
        upper=tuple(float(v) for v in interval.iloc[:, 1]),
    )


def _median_ci_from(fitter: Any) -> tuple[float | None, float | None]:
    from lifelines.utils import median_survival_times

    median_ci = median_survival_times(fitter.confidence_interval_)
    return (_finite_or_none(median_ci.iloc[0, 0]), _finite_or_none(median_ci.iloc[0, 1]))

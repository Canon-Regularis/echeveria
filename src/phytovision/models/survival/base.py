"""Contracts for survival analysis of time-to-wilt, with no heavy dependency at import time.

A survival model answers a different question from the forecaster: not what the score will be at a
horizon, but how long until the plant crosses the stressed cut, allowing for plants that never cross
inside the observed window (right censoring). The data types here are plain dataclasses over Python
lists, so this module imports with the base dependencies alone; lifelines and pandas are imported
lazily inside the concrete models, exactly as the state-space forecaster imports statsmodels.

Every estimate is synthetic-trained and RGB-proxy: see ``SURVIVAL_DISCLAIMER``. A median and its
band are indicative, not a validated prognosis.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Self

SURVIVAL_DISCLAIMER = (
    "survival estimates are synthetic-trained on the dry-down simulator; the covariates are RGB "
    "proxies (early stress level and slope); a median and its band are indicative, not a validated "
    "prognosis; right-censored sequences contribute only their observed time"
)


@dataclass(frozen=True, slots=True)
class SurvivalRecord:
    """One plant reduced to survival form: a 1-based duration, an event flag, and its covariates.

    ``duration`` is the number of observations up to and including the crossing (so it is at least
    one, which the parametric fitters require), and ``event_observed`` is 1 when the plant crossed
    the stressed cut inside the window, 0 when it was still below at the last observation.
    """

    plant_id: str
    duration: int
    event_observed: int
    covariates: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SurvivalDataset:
    """A cohort in survival form, ordered so durations, events, and covariates stay aligned."""

    records: tuple[SurvivalRecord, ...]

    def plant_ids(self) -> list[str]:
        return [record.plant_id for record in self.records]

    def durations(self) -> list[int]:
        return [record.duration for record in self.records]

    def events(self) -> list[int]:
        return [record.event_observed for record in self.records]

    @property
    def n_events(self) -> int:
        return sum(record.event_observed for record in self.records)

    def covariate_names(self) -> tuple[str, ...]:
        return tuple(self.records[0].covariates) if self.records else ()

    def covariate_frame(self) -> list[dict[str, float]]:
        return [dict(record.covariates) for record in self.records]

    def subset(self, indices: list[int]) -> SurvivalDataset:
        return SurvivalDataset(tuple(self.records[i] for i in indices))

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class SurvivalCurve:
    """A survival function over time, with an optional pointwise confidence band."""

    times: tuple[float, ...]
    survival: tuple[float, ...]
    lower: tuple[float, ...] = ()
    upper: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PlantSurvival:
    """One plant's time-to-wilt estimate. ``basis`` records which model produced it.

    ``median`` and the band are in observation steps, or None when no median falls inside the
    observed window. ``basis`` is ``cohort-km`` when the estimate is the cohort baseline broadcast,
    so a broadcast median is never mistaken for a per-plant covariate prediction.
    """

    plant_id: str
    median: float | None
    lower: float | None
    upper: float | None
    basis: str
    curve: SurvivalCurve | None = None


@dataclass(frozen=True, slots=True)
class SurvivalFit:
    """The result of fitting a cohort: the shared curve, the summary, and per-plant estimates."""

    model_name: str
    curve: SurvivalCurve
    cohort_median: float | None
    cohort_median_ci: tuple[float | None, float | None]
    concordance_index: float | None
    per_plant: dict[str, PlantSurvival]
    note: str
    disclaimer: str = SURVIVAL_DISCLAIMER


class SurvivalModel(ABC):
    """A survival model over a cohort. Concrete models add lifelines lazily inside fit/predict."""

    name: ClassVar[str] = "survival-model"
    note: ClassVar[str] = "synthetic-trained survival estimate; indicative, not a prognosis"
    uses_covariates: ClassVar[bool] = False

    @abstractmethod
    def fit(self, dataset: SurvivalDataset) -> Self:
        """Fit the model to a cohort's durations, events, and covariates."""

    @abstractmethod
    def predict(self, dataset: SurvivalDataset) -> dict[str, PlantSurvival]:
        """Estimate each plant's time-to-wilt from the fitted model."""

    def concordance_index(self, dataset: SurvivalDataset) -> float | None:
        """Held-out-capable concordance of the predicted medians against the observed times.

        A larger predicted median means longer survival, so it is passed to lifelines without
        negation. Medians that are None or infinite become a finite sentinel above every finite
        median, which ranks a plant with no in-window median as the longest-surviving. Returns None
        when the cohort has no events, or when no comparable pair exists.
        """
        if dataset.n_events == 0:
            return None
        try:
            from lifelines.utils import concordance_index as lifelines_concordance
        except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
            raise ImportError(_EXTRA_HINT) from exc

        durations = dataset.durations()
        events = dataset.events()
        predictions = self.predict(dataset)
        medians = {pid: _finite_or_none(predictions[pid].median) for pid in dataset.plant_ids()}
        finite = [m for m in medians.values() if m is not None]
        # The sentinel for a plant with no in-window median must outrank every actual predicted
        # median (some extrapolate past the longest duration), so a no-median plant reads as the
        # longest-surviving rather than shorter than a finite extrapolated median.
        sentinel = float(max([*durations, *finite]) + 1)
        scores = [sentinel if medians[pid] is None else medians[pid] for pid in dataset.plant_ids()]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(lifelines_concordance(durations, scores, events))
        except (ZeroDivisionError, ValueError):
            return None  # no admissible pair (for example every score is equal)


_EXTRA_HINT = 'the survival models need the stats extra: pip install -e ".[stats]"'


def _finite_or_none(value: object) -> float | None:
    """Coerce a value to a finite float, mapping None, NaN, and infinity to None."""
    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

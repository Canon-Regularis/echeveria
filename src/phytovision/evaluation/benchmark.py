"""Benchmark the registered forecasters over a cohort with time-series cross-validation.

Every forecaster runs on the same expanding-window origins over every plant, so the comparison is
apples to apples. Each forecast is scored against the held-out future with the probabilistic
metrics, aggregated per forecaster and per horizon, and reported as a mean with a confidence
interval, the shape ``CrossValResult`` uses for classification. A forecaster whose optional extra is
missing is skipped and named in ``skipped`` rather than silently dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from phytovision.evaluation._aggregate import mean_ci95
from phytovision.evaluation.probabilistic import (
    crps_gaussian_samples,
    interval_coverage,
    interval_pinball,
    mean_interval_width,
    std_from_interval,
)
from phytovision.evaluation.timeseries import expanding_window_splits
from phytovision.exceptions import ConfigError
from phytovision.registries import FORECASTERS
from phytovision.temporal import DEFAULT_HORIZONS, DEFAULT_INTERVAL_LEVEL, FeatureHistory

logger = logging.getLogger(__name__)

# The minimum training points before the first forecast origin. Two is the least a line can fit;
# four gives the confidence heuristic something to weigh.
_DEFAULT_MIN_TRAIN = 4


@dataclass(frozen=True, slots=True)
class ForecasterScore:
    """One forecaster's probabilistic accuracy at one horizon, aggregated over the cohort."""

    name: str
    horizon: int
    crps: float
    crps_ci95: tuple[float, float]
    pinball: float
    coverage: float
    mean_width: float
    n: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Every forecaster's per-horizon scores, plus any skipped for a missing extra."""

    scores: tuple[ForecasterScore, ...]
    interval_level: float
    skipped: tuple[str, ...] = ()

    def horizons(self) -> list[int]:
        return sorted({score.horizon for score in self.scores})

    def for_horizon(self, horizon: int) -> list[ForecasterScore]:
        """The scores at one horizon, ranked best CRPS first."""
        at_horizon = [score for score in self.scores if score.horizon == horizon]
        return sorted(at_horizon, key=lambda score: score.crps)

    def table(self) -> list[dict[str, object]]:
        """A flat, sorted table for printing or writing, ranked within each horizon."""
        rows: list[dict[str, object]] = []
        for horizon in self.horizons():
            for score in self.for_horizon(horizon):
                rows.append(
                    {
                        "horizon": score.horizon,
                        "forecaster": score.name,
                        "crps": round(score.crps, 5),
                        "crps_lo": round(score.crps_ci95[0], 5),
                        "crps_hi": round(score.crps_ci95[1], 5),
                        "pinball": round(score.pinball, 5),
                        "coverage": round(score.coverage, 4),
                        "mean_width": round(score.mean_width, 4),
                        "n": score.n,
                    }
                )
        return rows


class _Accumulator:
    """Collects held-out (actual, mean, lower, upper) tuples per horizon for one forecaster."""

    def __init__(self, horizons: Sequence[int]) -> None:
        self._data: dict[int, dict[str, list[float]]] = {
            h: {"actual": [], "mean": [], "lower": [], "upper": []} for h in horizons
        }

    def add(self, horizon: int, actual: float, mean: float, lower: float, upper: float) -> None:
        bucket = self._data[horizon]
        bucket["actual"].append(actual)
        bucket["mean"].append(mean)
        bucket["lower"].append(lower)
        bucket["upper"].append(upper)

    def score(self, name: str, interval_level: float) -> list[ForecasterScore]:
        scores: list[ForecasterScore] = []
        for horizon, bucket in self._data.items():
            if not bucket["actual"]:
                continue
            scores.append(_score_horizon(name, horizon, bucket, interval_level))
        return scores


def benchmark_forecasters(
    history: FeatureHistory,
    forecaster_names: Sequence[str] | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    min_train: int = _DEFAULT_MIN_TRAIN,
    interval_level: float = DEFAULT_INTERVAL_LEVEL,
) -> BenchmarkResult:
    """Run every named forecaster over the cohort's expanding-window origins and score each."""
    names = list(forecaster_names) if forecaster_names is not None else FORECASTERS.names()
    steps = sorted({h for h in horizons if h > 0})
    if not steps:
        raise ConfigError("benchmark needs at least one positive horizon")
    series_by_plant = {pid: _scores(history, pid) for pid in history.plant_ids}

    scores: list[ForecasterScore] = []
    skipped: list[str] = []
    for name in names:
        forecaster = FORECASTERS.create(name)
        accumulator = _Accumulator(steps)
        try:
            _run_forecaster(forecaster, series_by_plant, steps, min_train, accumulator)
        except ImportError as exc:
            logger.warning("skipping forecaster %s: %s", name, exc)
            skipped.append(name)
            continue
        scores.extend(accumulator.score(name, interval_level))
    return BenchmarkResult(tuple(scores), interval_level, tuple(skipped))


def _run_forecaster(
    forecaster: object,
    series_by_plant: dict[str, list[float]],
    steps: Sequence[int],
    min_train: int,
    accumulator: _Accumulator,
) -> None:
    max_horizon = max(steps)
    for series in series_by_plant.values():
        for train_index, _ in expanding_window_splits(len(series), min_train, max_horizon):
            origin = len(train_index)
            forecast = forecaster.forecast([series[i] for i in train_index], steps)  # type: ignore[attr-defined]
            for horizon in steps:
                actual_index = origin + horizon - 1
                if actual_index >= len(series):
                    continue
                mean = forecast.projected_scores[horizon]
                accumulator.add(
                    horizon,
                    series[actual_index],
                    mean,
                    forecast.lower.get(horizon, mean),
                    forecast.upper.get(horizon, mean),
                )


def _score_horizon(
    name: str, horizon: int, bucket: dict[str, list[float]], interval_level: float
) -> ForecasterScore:
    actual = bucket["actual"]
    mean = bucket["mean"]
    lower = bucket["lower"]
    upper = bucket["upper"]
    stds = std_from_interval(lower, upper, interval_level)
    samples = crps_gaussian_samples(actual, mean, stds)
    return ForecasterScore(
        name=name,
        horizon=horizon,
        crps=float(np.mean(samples)),
        crps_ci95=mean_ci95(samples),
        pinball=interval_pinball(actual, mean, lower, upper, interval_level),
        coverage=interval_coverage(actual, lower, upper),
        mean_width=mean_interval_width(lower, upper),
        n=len(actual),
    )


def _scores(history: FeatureHistory, plant_id: str) -> list[float]:
    return [observation.stress_score for observation in history.series_for(plant_id)]

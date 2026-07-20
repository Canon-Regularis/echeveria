"""Held-out evaluation of the survival models: a concordance-index leaderboard over a cohort.

The concordance index on the surface is in-sample and optimistic, because the model is fitted and
scored on the same cohort. This is the honest counterpart: k-fold over plant ids, fit each covariate
model on the training plants, and score its discrimination on the held-out plants, aggregated to a
mean with a confidence interval, the shape the forecaster benchmark reports. A model that could not
be scored, because its extra is missing or no fold had enough events, is named in ``skipped`` rather
than silently dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from phytovision.evaluation._aggregate import mean_ci95
from phytovision.exceptions import ConfigError
from phytovision.models.survival.base import SurvivalDataset, _finite_or_none
from phytovision.models.survival.cohort import derive_records
from phytovision.registries import SURVIVAL_MODELS
from phytovision.temporal.history import FeatureHistory

logger = logging.getLogger(__name__)

_MIN_EVENTS = 2


@dataclass(frozen=True, slots=True)
class SurvivalScore:
    """One survival model's held-out discrimination, aggregated over the folds."""

    name: str
    c_index: float
    c_index_ci95: tuple[float, float]
    folds: int


@dataclass(frozen=True, slots=True)
class SurvivalLeaderboard:
    """Every covariate model's held-out concordance, plus any that could not be scored (``skipped``:
    a missing extra, or no fold with enough events)."""

    scores: tuple[SurvivalScore, ...]
    n_folds: int
    skipped: tuple[str, ...] = ()

    def ranked(self) -> list[SurvivalScore]:
        """Scores ranked by concordance, best first."""
        return sorted(self.scores, key=lambda score: score.c_index, reverse=True)

    def table(self) -> list[dict[str, object]]:
        return [
            {
                "survival_model": score.name,
                "c_index": round(score.c_index, 4),
                "c_index_lo": round(score.c_index_ci95[0], 4),
                "c_index_hi": round(score.c_index_ci95[1], 4),
                "folds": score.folds,
            }
            for score in self.ranked()
        ]


def survival_concordance(
    medians: Mapping[str, float | None], dataset: SurvivalDataset
) -> float | None:
    """Concordance of predicted medians against observed durations, in dataset order.

    A larger median means longer survival, so it is passed without negation. A missing or infinite
    median becomes a finite sentinel just past the longest duration. Returns None with no events.
    """
    if dataset.n_events == 0:
        return None
    try:
        from lifelines.utils import concordance_index
    except ImportError as exc:  # pragma: no cover - depends on the optional stats extra
        hint = 'survival scoring needs the stats extra: pip install -e ".[stats]"'
        raise ImportError(hint) from exc

    durations = dataset.durations()
    resolved = {pid: _finite_or_none(medians.get(pid)) for pid in dataset.plant_ids()}
    finite = [m for m in resolved.values() if m is not None]
    # The sentinel must outrank every finite median (some extrapolate past the longest duration), so
    # a plant with no in-window median reads as longest-surviving, not shorter than a finite one.
    sentinel = float(max([*durations, *finite]) + 1)
    scores = [sentinel if resolved[pid] is None else resolved[pid] for pid in dataset.plant_ids()]
    try:
        return float(concordance_index(durations, scores, dataset.events()))
    except (ZeroDivisionError, ValueError):
        return None


def benchmark_survival_models(
    history: FeatureHistory,
    names: Sequence[str] | None = None,
    folds: int = 5,
    seed: int = 0,
    min_events: int = _MIN_EVENTS,
) -> SurvivalLeaderboard:
    """Rank the covariate survival models by held-out concordance over k folds of plants."""
    dataset = derive_records(history)
    if len(dataset) < 4:
        raise ConfigError("the survival benchmark needs at least four plants")
    chosen = list(names) if names is not None else _covariate_model_names()
    splits = _kfold(len(dataset), min(folds, len(dataset)), seed)

    scores: list[SurvivalScore] = []
    skipped: list[str] = []
    for name in chosen:
        fold_scores = _score_model(name, dataset, splits, min_events)
        if fold_scores is None:
            logger.warning("skipping survival model %s: needs an extra that is not installed", name)
            skipped.append(name)
        elif fold_scores:
            mean = float(np.mean(fold_scores))
            scores.append(SurvivalScore(name, mean, mean_ci95(fold_scores), len(fold_scores)))
        else:  # no fold had enough events to score: surface it rather than dropping it silently
            logger.warning("skipping survival model %s: no fold had enough events to score", name)
            skipped.append(name)
    return SurvivalLeaderboard(tuple(scores), len(splits), tuple(skipped))


def _score_model(
    name: str, dataset: SurvivalDataset, splits: list[tuple[list[int], list[int]]], min_events: int
) -> list[float] | None:
    """Held-out concordance per fold for one model, or None if its extra is missing."""
    fold_scores: list[float] = []
    for train_index, test_index in splits:
        train = dataset.subset(train_index)
        test = dataset.subset(test_index)
        if train.n_events < min_events or test.n_events < min_events:
            continue
        model = SURVIVAL_MODELS.create(name)
        try:
            model.fit(train)
        except ImportError:
            return None
        score = model.concordance_index(test)
        if score is not None:
            fold_scores.append(score)
    return fold_scores


def _covariate_model_names() -> list[str]:
    names = SURVIVAL_MODELS.names()
    return [name for name in names if SURVIVAL_MODELS.create(name).uses_covariates]


def _kfold(n: int, folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Deterministic k-fold split of ``range(n)`` into (train, test) index lists."""
    if folds < 2:
        raise ConfigError("the survival benchmark needs at least two folds")
    order = np.random.default_rng(seed).permutation(n)
    chunks = [chunk.tolist() for chunk in np.array_split(order, folds)]
    splits: list[tuple[list[int], list[int]]] = []
    for held_out in range(folds):
        test = chunks[held_out]
        train = [i for other, chunk in enumerate(chunks) if other != held_out for i in chunk]
        splits.append((train, test))
    return splits

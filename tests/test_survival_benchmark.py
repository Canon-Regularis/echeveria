"""The held-out survival leaderboard and the standalone concordance scorer."""

from __future__ import annotations

import importlib.util

import pytest

from phytovision.evaluation._aggregate import mean_ci95
from phytovision.evaluation.survival import (
    _covariate_model_names,
    _kfold,
    benchmark_survival_models,
    survival_concordance,
)
from phytovision.exceptions import ConfigError
from phytovision.models.survival import WeibullAFTSurvival, derive_records
from phytovision.models.survival.base import PlantSurvival, SurvivalDataset, SurvivalModel
from phytovision.registries import SURVIVAL_MODELS
from phytovision.simulation import DryDownParams, cohort_history, simulate_cohort

_HAS_LIFELINES = importlib.util.find_spec("lifelines") is not None
_needs_lifelines = pytest.mark.skipif(
    not _HAS_LIFELINES, reason="needs the stats extra (lifelines)"
)


class _MissingExtraSurvival(SurvivalModel):
    name = "benchmark-missing-survival"
    uses_covariates = True

    def fit(self, dataset: SurvivalDataset) -> _MissingExtraSurvival:
        raise ImportError("needs an extra that is not installed")

    def predict(self, dataset: SurvivalDataset) -> dict[str, PlantSurvival]:
        return {}


if "benchmark-missing-survival" not in SURVIVAL_MODELS:
    SURVIVAL_MODELS.register("benchmark-missing-survival")(_MissingExtraSurvival)


def _history():
    params = DryDownParams(n_steps=20, base_decline_rate=0.16, decline_rate_spread=0.5)
    return cohort_history(simulate_cohort(60, params, seed=7))


def test_kfold_folds_are_disjoint_and_cover_every_plant() -> None:
    splits = _kfold(20, folds=5, seed=0)
    assert len(splits) == 5
    for train, test in splits:
        assert not (set(train) & set(test))  # no plant in both train and test
        assert set(train) | set(test) == set(range(20))
    tested = {index for _, test in splits for index in test}
    assert tested == set(range(20))  # every plant is held out exactly once


@_needs_lifelines
def test_leaderboard_ranks_covariate_models_above_chance() -> None:
    # Explicit names, so a throwaway model another test registers does not enter the default set.
    board = benchmark_survival_models(_history(), names=["weibull-aft", "cox-ph"], folds=5, seed=0)
    assert board.skipped == ()
    assert {score.name for score in board.scores} == {"weibull-aft", "cox-ph"}
    for score in board.scores:
        assert score.c_index > 0.5  # held-out discrimination beats chance
        low, high = score.c_index_ci95
        assert abs((low + high) / 2 - score.c_index) < 1e-9  # the CI is centred on the mean
        assert score.folds >= 1
    assert board.ranked()[0].c_index == max(score.c_index for score in board.scores)


def test_ci95_kfold_and_benchmark_guards() -> None:
    assert mean_ci95([0.7]) == (0.7, 0.7)  # a single fold has a degenerate interval
    low, high = mean_ci95([0.6, 0.8])
    assert high > low and abs((low + high) / 2 - 0.7) < 1e-9
    with pytest.raises(ConfigError):
        _kfold(10, folds=1, seed=0)  # a benchmark needs at least two folds
    with pytest.raises(ConfigError):
        # Three plants is too few to hold any out; this raises before touching lifelines.
        benchmark_survival_models(
            cohort_history(simulate_cohort(3, DryDownParams(n_steps=8), seed=1))
        )


def test_covariate_model_names_excludes_the_baseline() -> None:
    names = set(_covariate_model_names())
    assert {"weibull-aft", "cox-ph"} <= names
    assert "kaplan-meier" not in names  # a covariate-free model has no per-plant ranking


@_needs_lifelines
def test_leaderboard_skips_a_model_whose_extra_is_missing() -> None:
    board = benchmark_survival_models(
        _history(), names=["weibull-aft", "benchmark-missing-survival"], folds=4, seed=0
    )
    assert board.skipped == ("benchmark-missing-survival",)
    assert {score.name for score in board.scores} == {"weibull-aft"}


@_needs_lifelines
def test_survival_concordance_scores_predicted_medians() -> None:
    history = _history()
    dataset = derive_records(history)
    fitted = WeibullAFTSurvival().fit(dataset)
    medians = {pid: plant.median for pid, plant in fitted.predict(dataset).items()}
    score = survival_concordance(medians, dataset)
    assert score is not None and score >= 0.55

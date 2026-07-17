"""Split-conformal regression intervals and the time-series splitter."""

from __future__ import annotations

import pytest

from phytovision.evaluation.conformal_regression import (
    ConformalIntervals,
    conformal_residual_quantile,
)
from phytovision.evaluation.timeseries import expanding_window_splits
from phytovision.exceptions import ConfigError
from phytovision.registries import FORECASTERS
from phytovision.simulation import DryDownParams, simulate_cohort


def test_residual_quantile_is_the_calibration_residual() -> None:
    # With ten residuals and alpha 0.1, k = ceil(11 * 0.9) = 10, the largest absolute residual.
    predictions = [0.0] * 10
    actuals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    qhat = conformal_residual_quantile(predictions, actuals, alpha=0.1)
    assert qhat == pytest.approx(0.45)


def test_interval_is_symmetric_and_clipped() -> None:
    intervals = ConformalIntervals(qhat=0.2, alpha=0.1)
    assert intervals.interval(0.5) == pytest.approx((0.3, 0.7))
    assert intervals.interval(0.95) == pytest.approx((0.75, 1.0))  # clipped at the top
    assert intervals.interval(0.05) == pytest.approx((0.0, 0.25))  # clipped at the bottom


def test_calibrate_rejects_mismatched_lengths() -> None:
    with pytest.raises(ConfigError):
        ConformalIntervals.calibrate([0.1, 0.2], [0.1], alpha=0.1)


def test_conformal_interval_reaches_nominal_coverage_on_the_simulator() -> None:
    # Pool one-step residuals of the linear forecaster over many origins, split into calibration and
    # test halves, and confirm the distribution-free interval covers near the nominal 0.9.
    cohort = simulate_cohort(60, DryDownParams(n_steps=14, base_decline_rate=0.10), seed=21)
    forecaster = FORECASTERS.create("linear-trend")
    predictions: list[float] = []
    actuals: list[float] = []
    for plant in cohort.series:
        scores = [obs.stress_score for obs in plant.observations]
        for cut in range(4, len(scores) - 1):
            predictions.append(forecaster.forecast(scores[:cut], (1,)).projected_scores[1])
            actuals.append(scores[cut])

    half = len(predictions) // 2
    intervals = ConformalIntervals.calibrate(predictions[:half], actuals[:half], alpha=0.1)
    covered = sum(
        intervals.interval(pred)[0] <= actual <= intervals.interval(pred)[1]
        for pred, actual in zip(predictions[half:], actuals[half:], strict=True)
    )
    coverage = covered / len(predictions[half:])
    assert coverage >= 0.85  # distribution-free, so close to the nominal 0.9


def test_expanding_window_never_leaks_the_future() -> None:
    splits = expanding_window_splits(10, min_train=3, horizon=2)
    assert splits  # non-empty
    for train, test in splits:
        assert train == list(range(len(train)))  # a growing prefix
        assert max(train) < min(test)  # no leakage
    # Every reachable future index is tested at least once.
    tested = {index for _, test in splits for index in test}
    assert tested == set(range(3, 10))


def test_expanding_window_grows_the_training_set() -> None:
    splits = expanding_window_splits(8, min_train=4, horizon=1, step=1)
    train_sizes = [len(train) for train, _ in splits]
    assert train_sizes == [4, 5, 6, 7]


def test_expanding_window_rejects_a_tiny_min_train() -> None:
    with pytest.raises(ConfigError):
        expanding_window_splits(10, min_train=1)

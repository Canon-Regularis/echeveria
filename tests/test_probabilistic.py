"""Probabilistic scoring rules: exact values on hand-built inputs, plus their guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phytovision.evaluation.probabilistic import (
    crps_gaussian,
    interval_coverage,
    interval_pinball,
    interval_score,
    mean_interval_width,
    pinball_loss,
    pit_values,
    std_from_interval,
)
from phytovision.exceptions import ContractViolationError


def test_crps_of_a_forecast_at_its_own_mean() -> None:
    # For a Gaussian centred on the outcome, CRPS = sigma * (sqrt(2/pi) - 1/sqrt(pi)).
    expected = math.sqrt(2.0 / math.pi) - 1.0 / math.sqrt(math.pi)
    assert crps_gaussian([0.0], [0.0], [1.0]) == pytest.approx(expected, abs=1e-6)


def test_crps_grows_as_the_mean_moves_away() -> None:
    near = crps_gaussian([0.0], [0.1], [0.2])
    far = crps_gaussian([0.0], [0.6], [0.2])
    assert far > near


def test_pinball_loss_is_asymmetric_by_quantile() -> None:
    # Under-prediction (y above q) is penalised by tau; over-prediction by (1 - tau).
    assert pinball_loss([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert pinball_loss([0.0], [1.0], 0.9) == pytest.approx(0.1)
    assert pinball_loss([1.0], [0.0], 0.5) == pytest.approx(0.5)


def test_pinball_rejects_a_degenerate_quantile() -> None:
    with pytest.raises(ContractViolationError):
        pinball_loss([0.5], [0.5], 0.0)


def test_interval_coverage_and_width() -> None:
    lower = [0.0, 0.2, 0.5]
    upper = [1.0, 0.4, 0.9]
    actuals = [0.5, 0.3, 0.95]  # inside, inside, outside
    assert interval_coverage(actuals, lower, upper) == pytest.approx(2.0 / 3.0)
    assert mean_interval_width(lower, upper) == pytest.approx((1.0 + 0.2 + 0.4) / 3.0)


def test_std_recovered_from_a_symmetric_interval() -> None:
    # A 90% interval of half-width z gives a unit standard deviation.
    z = 1.6448536269514722
    std = std_from_interval([0.0], [2.0 * z], 0.9)
    assert float(std[0]) == pytest.approx(1.0, abs=1e-6)


def test_pit_of_the_mean_is_one_half() -> None:
    pit = pit_values([0.3], [0.3], [0.2])
    assert float(pit[0]) == pytest.approx(0.5, abs=1e-9)


def test_interval_score_bundles_the_metrics() -> None:
    score = interval_score([0.5, 0.4], [0.5, 0.5], [0.3, 0.3], [0.7, 0.7], 0.9)
    assert score.n == 2
    assert score.coverage == pytest.approx(1.0)  # both inside [0.3, 0.7]
    assert score.mean_width == pytest.approx(0.4)
    assert score.crps >= 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ContractViolationError):
        crps_gaussian([0.0, 0.1], [0.0], [1.0])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ContractViolationError):
        interval_coverage([], [], [])


def test_metrics_are_finite_on_a_random_sample() -> None:
    rng = np.random.default_rng(0)
    actuals = rng.uniform(0, 1, 50)
    means = rng.uniform(0, 1, 50)
    lower = np.minimum(means, actuals) - 0.05
    upper = np.maximum(means, actuals) + 0.05
    stds = std_from_interval(lower, upper, 0.9)
    assert math.isfinite(crps_gaussian(actuals, means, stds))
    assert math.isfinite(interval_pinball(actuals, means, lower, upper, 0.9))

"""Calibration diagnostics for the stress score."""

from __future__ import annotations

import math

import pytest

from phytovision.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from phytovision.exceptions import ContractViolationError


def test_brier_score_is_zero_for_perfect_predictions() -> None:
    assert brier_score([0.0, 1.0, 0.0], [0, 1, 0]) == 0.0


def test_brier_score_is_one_for_confidently_wrong() -> None:
    assert brier_score([1.0], [0]) == 1.0


def test_reliability_curve_bins_and_summarises() -> None:
    curve = reliability_curve([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=2)
    assert curve.counts == (2, 2)
    assert curve.observed_rate[0] == 0.0
    assert curve.observed_rate[1] == 1.0


def test_expected_calibration_error_is_zero_when_calibrated() -> None:
    assert expected_calibration_error([0.0, 0.0, 1.0, 1.0], [0, 0, 1, 1], n_bins=2) == 0.0


def test_expected_calibration_error_is_size_weighted() -> None:
    # ECE weights each bin's gap by its share of observations. The existing test uses a perfectly
    # calibrated input returning 0.0, so the count/total weight is never exercised; here dropping it
    # (a plain mean of the two bin gaps) gives 0.5 instead of the size-weighted 0.3.
    ece = expected_calibration_error([0.1, 0.1, 0.1, 0.9], [0, 0, 0, 0], n_bins=2)
    assert ece == pytest.approx(0.3)  # (3/4)*0.1 + (1/4)*0.9, not the unweighted 0.5


def test_reliability_curve_reports_an_empty_bin_and_the_upper_edge() -> None:
    # A score exactly on the 0.5 edge falls in the upper bin (np.digitize right=False), leaving the
    # lower bin empty with NaN summaries and a zero count. right=True would flip the counts.
    curve = reliability_curve([0.5], [1], n_bins=2)
    assert curve.counts == (0, 1)
    assert math.isnan(curve.mean_score[0]) and math.isnan(curve.observed_rate[0])
    assert curve.mean_score[1] == pytest.approx(0.5)
    assert curve.observed_rate[1] == pytest.approx(1.0)


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ContractViolationError):
        brier_score([0.1], [0, 1])

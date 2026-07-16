"""Calibration diagnostics for the stress score."""

from __future__ import annotations

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


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ContractViolationError):
        brier_score([0.1], [0, 1])

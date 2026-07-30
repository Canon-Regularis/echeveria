"""Regression metrics for the score against a continuous target."""

from __future__ import annotations

import pytest

from phytovision.evaluation.regression import regression_metrics
from phytovision.exceptions import ContractViolationError


def test_perfect_prediction_scores_perfectly() -> None:
    metrics = regression_metrics([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert metrics.rmse == 0.0
    assert metrics.mae == 0.0
    assert metrics.r2 == 1.0


def test_a_constant_offset_gives_the_known_error() -> None:
    metrics = regression_metrics([0.1, 0.6, 1.1], [0.0, 0.5, 1.0])
    assert metrics.rmse == pytest.approx(0.1)
    assert metrics.mae == pytest.approx(0.1)


def test_r2_on_a_partial_fit_and_a_constant_target() -> None:
    # r2 = 1 - ss_res/ss_tot on a genuinely imperfect fit is 0.94, and it is 0.0 when the target has
    # no variance (ss_tot == 0), which the guard must return rather than dividing by zero. The
    # existing tests only cover the perfect fit (r2 == 1.0), so the formula and guard are unpinned.
    partial = regression_metrics([0.1, 0.6, 1.1], [0.0, 0.5, 1.0])
    assert partial.r2 == pytest.approx(0.94)

    constant = regression_metrics([0.1, 0.6], [0.3, 0.3])
    assert constant.r2 == 0.0  # ss_tot == 0 guard, not a ZeroDivisionError
    assert constant.rmse == pytest.approx(0.254951, abs=1e-6)


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ContractViolationError):
        regression_metrics([0.1], [0.1, 0.2])

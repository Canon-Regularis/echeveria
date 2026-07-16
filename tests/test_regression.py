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


def test_length_mismatch_is_rejected() -> None:
    with pytest.raises(ContractViolationError):
        regression_metrics([0.1], [0.1, 0.2])

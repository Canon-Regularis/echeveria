"""Binary metrics used by the evaluate command (F7)."""

from __future__ import annotations

import pytest

from phytovision.evaluation.metrics import binary_metrics


def test_counts_and_scores() -> None:
    # true: 1 1 0 0   pred: 1 0 0 1  -> tp=1, fn=1, tn=1, fp=1
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 0, 1])
    assert (metrics.tp, metrics.fn, metrics.tn, metrics.fp) == (1, 1, 1, 1)
    assert metrics.n == 4
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.f1 == 0.5


def test_perfect_prediction() -> None:
    metrics = binary_metrics([0, 1, 1], [0, 1, 1])
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_no_positives_is_safe() -> None:
    metrics = binary_metrics([0, 0], [0, 0])
    assert metrics.accuracy == 1.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        binary_metrics([1, 0], [1])

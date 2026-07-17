"""The shared normal-approximation confidence interval used across the evaluation harness."""

from __future__ import annotations

from phytovision.evaluation._aggregate import mean_ci95


def test_empty_input_is_zero() -> None:
    assert mean_ci95([]) == (0.0, 0.0)


def test_single_value_collapses_to_the_mean() -> None:
    assert mean_ci95([0.7]) == (0.7, 0.7)


def test_interval_is_centred_on_the_mean_with_positive_width() -> None:
    low, high = mean_ci95([0.6, 0.8])
    assert high > low
    assert abs((low + high) / 2 - 0.7) < 1e-9

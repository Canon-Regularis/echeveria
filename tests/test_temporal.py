"""Temporal tracking scaffold (F27): a per-plant feature-history store plus a stress-trend fit."""

from __future__ import annotations

import pytest

from phytovision.pipeline import Pipeline
from phytovision.temporal import FeatureHistory, Observation, stress_trend


def _obs(plant_id: str, timestamp: str, score: float) -> Observation:
    return Observation(plant_id=plant_id, timestamp=timestamp, stress_score=score)


def test_history_groups_and_orders_by_timestamp() -> None:
    history = FeatureHistory()
    # Add out of order to prove series_for sorts by timestamp, not insertion order.
    history.add(_obs("plant-a", "2026-03-02", 0.5))
    history.add(_obs("plant-a", "2026-03-01", 0.2))
    history.add(_obs("plant-b", "2026-03-01", 0.9))

    assert history.plant_ids == ["plant-a", "plant-b"]
    assert len(history) == 3
    timestamps = [obs.timestamp for obs in history.series_for("plant-a")]
    assert timestamps == ["2026-03-01", "2026-03-02"]
    assert history.series_for("unknown-plant") == []


def test_record_builds_an_observation_from_a_report(healthy_image) -> None:
    report = Pipeline.default().analyze(healthy_image)
    history = FeatureHistory()
    observation = history.record("plant-a", "2026-03-01", report)

    assert observation.stress_score == report.stress.score
    assert observation.features == report.plant_features.defined()
    assert history.series_for("plant-a") == [observation]


def _series(scores: list[float]) -> list[Observation]:
    return [_obs("p", f"2026-03-{i + 1:02d}", score) for i, score in enumerate(scores)]


def test_trend_detects_a_rising_series() -> None:
    trend = stress_trend("p", _series([0.1, 0.3, 0.6, 0.8]))
    assert trend.direction == "rising"
    assert trend.slope == pytest.approx(0.24)  # least-squares slope, pinned to catch a scaling bug
    assert trend.n == 4
    assert (trend.start_score, trend.end_score) == (0.1, 0.8)


def test_trend_detects_a_falling_series() -> None:
    trend = stress_trend("p", _series([0.9, 0.6, 0.3, 0.1]))
    assert trend.direction == "falling"
    assert trend.slope == pytest.approx(-0.27)


def test_trend_direction_brackets_the_dead_band() -> None:
    # Slope 0.02 is just above the 0.01 dead band (rising); 0.005 is just below it (flat). This pins
    # the threshold so it cannot silently widen and misclassify a genuine trend.
    assert stress_trend("p", _series([0.50, 0.52, 0.54, 0.56])).direction == "rising"
    assert stress_trend("p", _series([0.500, 0.505, 0.510, 0.515])).direction == "flat"
    assert stress_trend("p", _series([0.50, 0.51, 0.49, 0.50])).direction == "flat"


def test_trend_sorts_a_shuffled_series_before_fitting() -> None:
    # Feed observations out of chronological order; stress_trend must sort them so the start/end
    # scores and the rising verdict reflect time order, not argument order.
    shuffled = [
        _obs("p", "2026-03-03", 0.6),
        _obs("p", "2026-03-01", 0.1),
        _obs("p", "2026-03-04", 0.8),
        _obs("p", "2026-03-02", 0.3),
    ]
    trend = stress_trend("p", shuffled)
    assert trend.direction == "rising"
    assert (trend.start_score, trend.end_score) == (0.1, 0.8)


def test_trend_handles_empty_and_single_series() -> None:
    assert stress_trend("p", []).direction == "unknown"
    single = stress_trend("p", [_obs("p", "2026-03-01", 0.4)])
    assert single.direction == "flat"
    assert single.n == 1

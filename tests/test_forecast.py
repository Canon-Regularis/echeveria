"""Trajectory forecasting: linear extrapolation of the stress score to future horizons."""

from __future__ import annotations

from phytovision.temporal import FeatureHistory, Forecast, plant_forecasts, stress_forecast
from phytovision.temporal.history import Observation


def _obs(timestamp: str, score: float) -> Observation:
    return Observation("p", timestamp, score, {})


def test_rising_trajectory_projects_upward_and_reaches_stressed() -> None:
    series = [_obs("2026-03-01", 0.2), _obs("2026-03-02", 0.4), _obs("2026-03-03", 0.6)]
    forecast = stress_forecast("p", series, horizons=(1, 3))
    assert forecast.slope > 0
    assert forecast.projected_scores[3] >= forecast.projected_scores[1]  # further ahead, higher
    assert forecast.steps_to_stressed is not None and forecast.steps_to_stressed >= 1
    assert 0.0 <= forecast.confidence <= 1.0


def test_steps_to_stressed_matches_the_projection_under_noise() -> None:
    # A noisy final dip must not desync steps_to_stressed from the fitted projection: both anchor on
    # the fitted trend, so the projected score at the reported step is the first to cross the cut.
    series = [_obs(f"2026-03-0{i + 1}", s) for i, s in enumerate([0.2, 0.4, 0.6, 0.8, 0.3])]
    forecast = stress_forecast("p", series, horizons=(1, 2, 3))
    assert forecast.steps_to_stressed == 2
    assert forecast.projected_scores[2] >= 0.66  # crosses at the reported step
    assert forecast.projected_scores[1] < 0.66  # and not before it


def test_falling_trajectory_has_no_time_to_stressed() -> None:
    series = [_obs("2026-03-01", 0.6), _obs("2026-03-02", 0.4), _obs("2026-03-03", 0.2)]
    forecast = stress_forecast("p", series)
    assert forecast.slope < 0
    assert forecast.steps_to_stressed is None


def test_already_stressed_has_no_time_to_stressed() -> None:
    forecast = stress_forecast("p", [_obs("2026-03-01", 0.7), _obs("2026-03-02", 0.8)])
    assert forecast.steps_to_stressed is None  # latest score already past the stressed cut


def test_flat_trajectory_has_no_time_to_stressed() -> None:
    series = [_obs(f"2026-03-0{i + 1}", 0.3) for i in range(3)]
    forecast = stress_forecast("p", series)
    assert forecast.steps_to_stressed is None
    assert abs(forecast.slope) <= 0.011


def test_single_and_empty_series_are_low_confidence_flat() -> None:
    single = stress_forecast("p", [_obs("2026-03-01", 0.5)], horizons=(1, 3))
    assert single.steps_to_stressed is None
    assert single.confidence == 0.1
    assert single.projected_scores == {1: 0.5, 3: 0.5}
    empty = stress_forecast("p", [], horizons=(1,))
    assert empty.projected_scores == {1: 0.0}


def test_confidence_grows_with_length_and_decays_with_horizon() -> None:
    short = stress_forecast("p", [_obs("2026-03-01", 0.2), _obs("2026-03-02", 0.4)], horizons=(1,))
    long_series = [_obs(f"2026-03-0{i + 1}", 0.2 + 0.1 * i) for i in range(5)]
    long_near = stress_forecast("p", long_series, horizons=(1,))
    long_far = stress_forecast("p", long_series, horizons=(30,))
    assert long_near.confidence > short.confidence  # more observations, more confidence
    assert long_near.confidence > long_far.confidence  # nearer horizon, more confidence


def test_plant_forecasts_over_a_history() -> None:
    history = FeatureHistory()
    for i, score in enumerate([0.2, 0.4, 0.6]):
        history.add(_obs(f"2026-03-0{i + 1}", score))
    forecasts = plant_forecasts(history, horizons=(1,))
    assert set(forecasts) == {"p"}
    assert isinstance(forecasts["p"], Forecast)

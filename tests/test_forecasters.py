"""The pluggable trajectory forecasters: shape, intervals, crossing, coverage, and degradation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from phytovision.exceptions import ConfigError, ContractViolationError
from phytovision.models.forecasting.base import Prediction, SeriesForecaster
from phytovision.registries import FORECASTERS
from phytovision.simulation import DryDownParams, simulate_cohort
from phytovision.temporal import stress_forecast
from phytovision.temporal.history import Observation


def _available() -> list[str]:
    """Forecaster names runnable in this environment (statistical ones need their extras)."""
    names = ["linear-trend"]
    if importlib.util.find_spec("sklearn"):
        names += ["gaussian-process", "bayesian-ridge"]
    if importlib.util.find_spec("statsmodels"):
        names += ["state-space", "arima"]
    return names


def _rising_series() -> list[float]:
    return [0.15, 0.22, 0.31, 0.38, 0.47, 0.55, 0.61]


@pytest.mark.parametrize("name", _available())
def test_forecast_has_bracketed_intervals(name: str) -> None:
    forecast = FORECASTERS.create(name).forecast(_rising_series(), (1, 3, 7), "p")
    assert forecast.method == name
    assert set(forecast.projected_scores) == {1, 3, 7}
    for horizon in (1, 3, 7):
        mean = forecast.projected_scores[horizon]
        assert forecast.lower[horizon] <= mean <= forecast.upper[horizon]
        assert 0.0 <= forecast.lower[horizon] <= forecast.upper[horizon] <= 1.0


@pytest.mark.parametrize("name", _available())
def test_a_rising_series_projects_upward(name: str) -> None:
    forecast = FORECASTERS.create(name).forecast(_rising_series(), (1, 7), "p")
    assert forecast.projected_scores[7] >= forecast.projected_scores[1]


def test_linear_baseline_matches_the_point_forecast() -> None:
    # The baseline must not move the number: its projection equals the historical stress_forecast.
    series = [
        Observation("p", f"2026-03-{i + 1:02d}", s, {}) for i, s in enumerate(_rising_series())
    ]
    reference = stress_forecast("p", series, (1, 3, 7))
    baseline = FORECASTERS.create("linear-trend").forecast(_rising_series(), (1, 3, 7), "p")
    assert baseline.projected_scores == reference.projected_scores
    assert baseline.steps_to_stressed == reference.steps_to_stressed


def test_t_multiplier_widens_at_small_dof_and_converges_to_normal() -> None:
    from statistics import NormalDist

    from phytovision.temporal.forecast import t_multiplier

    z = NormalDist().inv_cdf(0.95)  # the two-sided normal multiplier at the 0.9 level
    assert t_multiplier(0.9, 3) > t_multiplier(0.9, 30) > z  # smaller n, wider band
    assert t_multiplier(0.9, 100000) == pytest.approx(z, abs=1e-3)  # converges to normal at large n
    assert t_multiplier(0.9, 0) == t_multiplier(0.9, 1)  # dof floored at one for a two-point fit


def test_small_sample_interval_is_wider_than_the_normal_one() -> None:
    from statistics import NormalDist

    from phytovision.temporal.forecast import linear_prediction_interval

    # A short series estimates the residual spread from few points, so the Student-t band is wider
    # than the normal quantile would give, which is what fixes the undercoverage at small n.
    scores = [0.20, 0.28, 0.24, 0.33]  # n = 4, so 2 residual degrees of freedom
    lower, upper = linear_prediction_interval(scores, (1,), level=0.9)
    t_width = upper[1] - lower[1]
    # The normal-quantile width at the same residual spread, computed the same way.
    from phytovision.temporal._fit import fit_line

    slope, intercept, _ = fit_line(scores)
    residuals = [y - (intercept + slope * x) for x, y in enumerate(scores)]
    resid_std = max(0.02, (sum(r * r for r in residuals) / 2) ** 0.5)
    mean_x, sxx = 1.5, sum((x - 1.5) ** 2 for x in range(4))
    leverage = 1.0 + 1.0 / 4 + (4 - mean_x) ** 2 / sxx
    z_width = 2 * NormalDist().inv_cdf(0.95) * resid_std * leverage**0.5
    assert t_width > z_width


def test_intervals_widen_with_horizon() -> None:
    # Further ahead is less certain, so the band is wider. A mid-range series keeps the band off the
    # [0, 1] clip, where widening would otherwise be hidden.
    midrange = [0.30, 0.32, 0.31, 0.33, 0.32, 0.34, 0.33]
    forecast = FORECASTERS.create("linear-trend").forecast(midrange, (1, 7), "p")
    near = forecast.upper[1] - forecast.lower[1]
    far = forecast.upper[7] - forecast.lower[7]
    assert far > near


@pytest.mark.parametrize("name", _available())
def test_interval_keeps_width_when_the_projection_passes_the_ceiling(name: str) -> None:
    # A steep series projects past the [0, 1] ceiling at the far horizon. Centring the band on the
    # clipped mean keeps a finite width there, so the probabilistic scorer is never handed a
    # zero-width interval it would read as near-certain.
    forecast = FORECASTERS.create(name).forecast([0.1, 0.3, 0.5, 0.7, 0.9], (1, 7), "p")
    assert forecast.projected_scores[7] == pytest.approx(1.0)  # clipped to the ceiling
    assert forecast.upper[7] - forecast.lower[7] > 0.0


def test_state_space_reader_recentres_on_the_clipped_mean() -> None:
    # forecast_with_intervals recentres the model's band on the clipped mean. Clipping the raw mean
    # and both interval columns independently would collapse a projection past the ceiling to a
    # zero-width [1.0, 1.0] band, so this covers the reader even when statsmodels is absent.
    from phytovision.models.forecasting.state_space import forecast_with_intervals

    mean_all = np.array([0.6, 0.8, 1.0, 1.4, 1.8, 2.1, 2.3])
    conf = np.column_stack([mean_all - 0.2, mean_all + 0.2])

    class _Forecast:
        def __init__(self, mean: np.ndarray, band: np.ndarray) -> None:
            self.predicted_mean = mean
            self._band = band

        def conf_int(self, alpha: float) -> np.ndarray:
            return self._band

    class _Result:
        def get_forecast(self, steps: int) -> _Forecast:
            return _Forecast(mean_all[:steps], conf[:steps])

    prediction = forecast_with_intervals(_Result(), [7], level=0.8)
    assert prediction.mean[7] == 1.0  # clipped to the ceiling
    assert prediction.upper[7] - prediction.lower[7] == pytest.approx(0.2)  # width preserved


@pytest.mark.parametrize("name", _available())
def test_short_series_degrades_to_a_flat_forecast(name: str) -> None:
    forecast = FORECASTERS.create(name).forecast([0.5], (1, 3), "p")
    assert forecast.projected_scores == {1: 0.5, 3: 0.5}
    assert forecast.steps_to_stressed is None
    assert forecast.confidence == 0.1


def test_already_stressed_reports_no_time_to_stressed() -> None:
    forecast = FORECASTERS.create("linear-trend").forecast([0.7, 0.75, 0.82], (1, 3), "p")
    assert forecast.steps_to_stressed is None


def test_linear_interval_brackets_held_out_truth_most_of_the_time() -> None:
    # Over a synthetic cohort the one-step interval should bracket the true next score much of the
    # time. The nominal level is 0.9, but the latent dry-down is a saturating curve, so a linear
    # model is slightly misspecified and honestly undercovers. It should still bracket most steps
    # and not be vacuously wide. The distribution-free conformal interval reaches nominal coverage;
    # that is asserted with the conformal regression tests.
    cohort = simulate_cohort(40, DryDownParams(n_steps=14, base_decline_rate=0.10), seed=11)
    forecaster = FORECASTERS.create("linear-trend")
    covered = 0
    widths: list[float] = []
    total = 0
    for plant in cohort.series:
        scores = [obs.stress_score for obs in plant.observations]
        for cut in range(4, len(scores) - 1):
            forecast = forecaster.forecast(scores[:cut], (1,), plant.plant_id)
            actual = scores[cut]  # the held-out next step
            if forecast.lower[1] <= actual <= forecast.upper[1]:
                covered += 1
            widths.append(forecast.upper[1] - forecast.lower[1])
            total += 1
    assert total > 0
    assert covered / total >= 0.6  # brackets most steps
    assert sum(widths) / len(widths) < 0.6  # yet stays sharp, not a vacuous [0, 1] band


def test_a_non_finite_score_is_rejected() -> None:
    from phytovision.exceptions import ContractViolationError
    from phytovision.models.forecasting.baseline import LinearTrendForecaster

    # A non-finite score silently projected a confident, tight 0.0; it is now rejected loudly.
    with pytest.raises(ContractViolationError):
        LinearTrendForecaster().forecast([0.2, 0.3, float("nan"), 0.5, 0.6], (1, 3, 7))


def test_an_interval_level_that_rounds_to_one_is_rejected() -> None:
    from phytovision.exceptions import ConfigError
    from phytovision.models.forecasting.baseline import LinearTrendForecaster
    from phytovision.temporal.forecast import forecast_scores

    # The largest double below 1.0 passes 0 < level < 1, yet (1 + level) / 2 rounds to exactly 1.0
    # and crashes NormalDist. It is now rejected at construction and at the free-function entry.
    edge = 0.9999999999999999
    with pytest.raises(ConfigError):
        LinearTrendForecaster(edge)
    with pytest.raises(ConfigError):
        forecast_scores("p", [0.1, 0.2, 0.3, 0.4], (1,), interval_level=edge)


def test_state_space_band_is_floored_not_absurdly_narrow() -> None:
    pytest.importorskip("statsmodels")
    from phytovision.models.forecasting.state_space import StateSpaceForecaster

    # A boundary-solution fit on a smooth series used to report a band ~1e-4 wide from six points;
    # the residual-std floor keeps it at least as wide as the other forecasters' minimum.
    forecast = StateSpaceForecaster().forecast([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (1, 3))
    assert forecast.upper[1] - forecast.lower[1] > 0.02


def test_bayesian_ridge_band_is_floored_not_absurdly_narrow() -> None:
    pytest.importorskip("sklearn")
    from phytovision.models.forecasting.bayesian import BayesianRidgeForecaster

    # A smooth interior series (no ceiling clipping) used to report a band tens of times too narrow;
    # flooring the spread keeps it at least as wide as the other forecasters' minimum.
    series = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37]
    forecast = BayesianRidgeForecaster().forecast(series, (1, 3, 7))
    assert forecast.upper[1] - forecast.lower[1] > 0.02


class _NumericFailure(SeriesForecaster):
    name = "boom"

    def _predict(self, scores, steps):  # type: ignore[no-untyped-def]
        raise ValueError("ill-conditioned")


class _MissingExtra(SeriesForecaster):
    name = "no-extra"

    def _predict(self, scores, steps):  # type: ignore[no-untyped-def]
        raise ImportError("needs the stats extra")


def test_a_numeric_failure_degrades_to_the_linear_interval() -> None:
    forecast = _NumericFailure().forecast([0.1, 0.2, 0.3, 0.4], (1, 3), "p")
    assert forecast.projected_scores  # produced by the linear fallback, not a crash
    assert forecast.lower and forecast.upper
    assert forecast.degraded is True  # the fallback is flagged so the benchmark can surface it


def test_a_successful_forecast_is_not_flagged_degraded() -> None:
    forecast = FORECASTERS.create("linear-trend").forecast(_rising_series(), (1, 3), "p")
    assert forecast.degraded is False


def test_a_missing_extra_raises_rather_than_silently_degrading() -> None:
    with pytest.raises(ImportError):
        _MissingExtra().forecast([0.1, 0.2, 0.3, 0.4], (1,), "p")


class _FlatSeriesForecaster(SeriesForecaster):
    """A minimal concrete SeriesForecaster (a flat mean at the last score) that exercises the shared
    template. LinearTrendForecaster overrides forecast(), so the base template's finite-guard and
    crossing search are only reachable through a subclass like this one."""

    name = "flat-test"

    def _predict(self, scores, steps):  # type: ignore[no-untyped-def]
        last = scores[-1]
        return Prediction(
            {s: last for s in steps},
            {s: max(0.0, last - 0.1) for s in steps},
            {s: min(1.0, last + 0.1) for s in steps},
        )


def test_series_forecaster_rejects_a_non_finite_score() -> None:
    # The shared forecast() finite-guard that the statistical forecasters (GP, ARIMA, state-space,
    # Bayesian) all rely on is only reachable through a SeriesForecaster subclass, since
    # LinearTrendForecaster overrides forecast() with its own separate guard. A NaN or inf must be
    # rejected loudly, not projected as a confident 0.0.
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ContractViolationError):
            _FlatSeriesForecaster().forecast([0.1, bad, 0.3], (1, 3), "p")


def test_series_forecaster_reports_no_crossing_for_a_falling_series() -> None:
    # A falling series never reaches the stressed cut, so the crossing search scans the whole
    # 1.._MAX_LOOKAHEAD window and returns None. That terminal path is bypassed by the linear
    # baseline, so only a SeriesForecaster subclass exercises it.
    forecast = _FlatSeriesForecaster().forecast([0.6, 0.5, 0.4, 0.3], (1, 3, 7), "p")
    assert forecast.steps_to_stressed is None


def test_series_forecaster_rejects_an_interval_level_that_rounds_to_one() -> None:
    # A level a hair below 1.0 whose (1 + level) / 2 rounds up to exactly 1.0 must be rejected at
    # construction, or the interval quantile crashes downstream.
    with pytest.raises(ConfigError):
        _FlatSeriesForecaster(interval_level=0.9999999999999999)

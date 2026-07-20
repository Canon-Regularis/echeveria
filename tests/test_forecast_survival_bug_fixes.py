"""Regression tests for a batch of forecasting and survival correctness fixes.

The Gaussian-process band widens with the horizon, a degenerate state-space forecast is coerced not
mislabelled, the survival covariate window is capped at the crossing, the concordance sentinel
outranks over-extrapolated medians, and a truncated manifest row fails cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.exceptions import ConfigError


def test_gaussian_process_interval_widens_with_the_horizon() -> None:
    pytest.importorskip("sklearn")
    from phytovision.models.forecasting.gaussian_process import GaussianProcessForecaster

    series = [0.30, 0.34, 0.39, 0.43, 0.48, 0.52, 0.57]  # a near-perfect line
    forecast = GaussianProcessForecaster().forecast(series, (1, 7))
    width_1 = forecast.upper[1] - forecast.lower[1]
    width_7 = forecast.upper[7] - forecast.lower[7]
    # The band now folds in the detrending line's extrapolation uncertainty, so it widens with the
    # horizon instead of staying near-zero (overconfident) far from the observed window.
    assert width_7 > width_1
    assert width_7 > 0.05


def test_forecast_with_intervals_coerces_a_degenerate_shape() -> None:
    from phytovision.models.forecasting.state_space import forecast_with_intervals

    class _Forecast:
        predicted_mean = np.array(0.5)  # 0-d, as a two-point fit can return

        def conf_int(self, alpha: float) -> np.ndarray:
            return np.array([0.4, 0.6])  # 1-d

    class _Result:
        def get_forecast(self, steps: int) -> _Forecast:
            return _Forecast()

    prediction = forecast_with_intervals(_Result(), (1,), 0.9)  # must not raise IndexError
    assert prediction.mean[1] == 0.5
    assert prediction.lower[1] == 0.4 and prediction.upper[1] == 0.6


def test_early_covariates_are_capped_at_the_crossing() -> None:
    from phytovision.models.survival.cohort import derive_records
    from phytovision.temporal import FeatureHistory
    from phytovision.temporal.history import Observation

    history = FeatureHistory()
    for step, score in enumerate([0.20, 0.70, 0.90, 0.90]):  # crosses at index 1
        history.add(
            Observation(plant_id="fast", timestamp=f"2024-01-{step + 1:02d}", stress_score=score)
        )
    record = next(r for r in derive_records(history).records if r.plant_id == "fast")
    # The window is capped to the pre-crossing observation, so the baseline is 0.20, not the mean of
    # the first three (0.60), which would leak the crossing into the "early" covariate.
    assert record.covariates["baseline_stress"] == 0.20


def test_concordance_sentinel_outranks_an_over_extrapolated_median() -> None:
    pytest.importorskip("lifelines")
    from phytovision.evaluation.survival import survival_concordance
    from phytovision.models.survival.base import SurvivalDataset, SurvivalRecord

    dataset = SurvivalDataset(
        (
            SurvivalRecord(plant_id="P1", duration=1, event_observed=1),
            SurvivalRecord(plant_id="P2", duration=3, event_observed=1),
            SurvivalRecord(plant_id="P3", duration=2, event_observed=1),
        )
    )
    # P2 has no in-window median (it survives longest); P3's finite median over-extrapolates past
    # the longest duration. The sentinel must sit above 50 so P2 still reads as longest-surviving,
    # which makes the ranking perfectly concordant with the durations.
    medians: dict[str, float | None] = {"P1": 1.0, "P2": None, "P3": 50.0}
    assert survival_concordance(medians, dataset) == 1.0


def test_load_history_rejects_a_truncated_row(tmp_path) -> None:
    from phytovision.simulation import load_history

    manifest = tmp_path / "cohort.csv"
    manifest.write_text(
        "plant_id,timestamp,stress_score,colour.gcc_mean\nplant_0,2024-01-01\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):  # a missing stress_score cell is None; float(None) would crash
        load_history(str(manifest))

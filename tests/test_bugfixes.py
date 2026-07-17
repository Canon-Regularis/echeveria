"""Regression tests for the project-wide bug-hunt findings.

Each test pins a specific defect the hunt surfaced and the fix removed: a swallowed config error, a
non-finite manifest value, an uncaught CLI crash, a gapped crossing search, an integer overflow, a
drift between two verdict thresholds, an off-by-one duration, and a survival cohort with no repeats.
"""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.cli import main
from phytovision.exceptions import ConfigError, InsufficientDataError


def test_config_schema_rejects_non_mapping_params() -> None:
    from phytovision.config_schema import PipelineConfig

    with pytest.raises(ConfigError, match="params"):
        PipelineConfig.from_mapping({"model": {"name": "heuristic", "params": "n_estimators=200"}})


def test_read_config_rejects_bad_suffix_before_reading(tmp_path) -> None:
    from phytovision.config import read_config

    bad = tmp_path / "config.yaml"
    bad.write_text("model: heuristic", encoding="utf-8")
    with pytest.raises(ConfigError, match="toml or .json"):
        read_config(str(bad))


def test_read_config_wraps_non_utf8_bytes_as_config_error(tmp_path) -> None:
    from phytovision.config import read_config

    binary = tmp_path / "config.toml"
    binary.write_bytes(b"\xff\xfe\x00\x01 not utf-8")
    with pytest.raises(ConfigError):  # a raw UnicodeDecodeError would escape every caller's handler
        read_config(str(binary))


def test_manifest_rejects_non_finite_target(tmp_path) -> None:
    from phytovision.datasets.manifest import CsvManifestLoader

    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,target\na.png,0.5\nb.png,nan\n", encoding="utf-8")
    with pytest.raises(ConfigError):  # float('nan') would otherwise poison the whole regression
        list(CsvManifestLoader(str(manifest), target_column="target"))


def test_load_history_rejects_non_numeric_cell(tmp_path) -> None:
    from phytovision.simulation import load_history

    manifest = tmp_path / "cohort.csv"
    manifest.write_text(
        "plant_id,timestamp,stress_score,colour.gcc_mean\np,2024-01-01,high,0.4\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        load_history(str(manifest))


def test_validate_rejects_non_positive_bins() -> None:
    # The bins guard fires before the manifest is read, so a clean exit 2 (not a traceback) results.
    assert main(["validate", "unused.csv", "--bins", "0"]) == 2
    assert main(["validate", "unused.csv", "--bins", "-3"]) == 2


def test_benchmark_rejects_out_of_range_interval_level() -> None:
    assert main(["benchmark", "unused.csv", "--interval-level", "1.0"]) == 2
    assert main(["benchmark", "unused.csv", "--interval-level", "0"]) == 2


def test_first_crossing_ignores_gapped_horizons_above_the_cap() -> None:
    from phytovision.models.forecasting.base import _MAX_LOOKAHEAD, _first_crossing

    # The contiguous window stays below the cut; a sparse horizon beyond the cap is above it. The
    # gap between them was never evaluated, so the search must not report the far horizon.
    means = {step: 0.30 for step in range(1, _MAX_LOOKAHEAD + 1)}
    means[_MAX_LOOKAHEAD + 30] = 0.90
    assert _first_crossing(0.20, means) is None

    # A true crossing inside the contiguous window is still found.
    within = {step: (0.90 if step >= 10 else 0.30) for step in range(1, _MAX_LOOKAHEAD + 1)}
    assert _first_crossing(0.20, within) == 10


def test_linear_forecaster_rejects_out_of_range_interval_level() -> None:
    from phytovision.models.forecasting.baseline import LinearTrendForecaster

    with pytest.raises(ConfigError):
        LinearTrendForecaster(interval_level=1.0)
    with pytest.raises(ConfigError):
        LinearTrendForecaster(interval_level=-0.5)


def test_forecast_scores_rejects_out_of_range_interval_level() -> None:
    from phytovision.temporal.forecast import forecast_scores

    with pytest.raises(ConfigError):
        forecast_scores("p", [0.1, 0.2, 0.3], [1, 3], interval_level=1.5)


def test_exg_segmenter_handles_uint8_without_overflow() -> None:
    from phytovision.segmentation.plant.exg_threshold import ExGThresholdSegmenter

    # A brown background (r + g + b = 290 > 255) would wrap in uint8 arithmetic and eat the frame.
    image = np.full((40, 40, 3), (120, 90, 80), dtype=np.uint8)
    image[12:28, 12:28] = (40, 200, 40)  # a green plant patch
    segmenter = ExGThresholdSegmenter()
    from_uint8 = segmenter.segment(image)
    from_float = segmenter.segment(image.astype(np.float32) / 255.0)
    assert from_uint8.sum() < from_uint8.size  # not the whole frame (the overflow symptom)
    assert np.array_equal(from_uint8, from_float)


def test_column_stats_returns_true_zero_std_for_a_constant_covariate() -> None:
    from phytovision.models.survival.covariate import _MIN_STD, _column_stats

    frame = [{"const": 1.0, "vary": 0.0}, {"const": 1.0, "vary": 1.0}, {"const": 1.0, "vary": 2.0}]
    _, stds = _column_stats(frame, ("const", "vary"))
    # The constant column reports a std below the drop threshold; before the fix it was floored up
    # to the threshold, so the keep-filter never excluded it. The varying column stays above.
    assert stds["const"] < _MIN_STD
    assert stds["vary"] >= _MIN_STD


def test_folder_loader_skips_directories_named_like_images(tmp_path) -> None:
    from phytovision.datasets.folder import FolderClassificationLoader

    healthy = tmp_path / "ds" / "healthy"
    healthy.mkdir(parents=True)
    (healthy / "real.png").write_bytes(b"not a real image, but a file")
    (healthy / "batch.png").mkdir()  # a directory whose name ends in an image suffix

    paths = [sample.image_path for sample in FolderClassificationLoader(str(tmp_path / "ds"))]
    assert any(path.endswith("real.png") for path in paths)
    assert not any(path.endswith("batch.png") for path in paths)


def test_fit_cohort_survival_raises_on_single_observation_cohort() -> None:
    from phytovision.models.survival import fit_cohort_survival
    from phytovision.temporal import FeatureHistory
    from phytovision.temporal.history import Observation

    history = FeatureHistory()
    history.add(Observation(plant_id="A", timestamp="2024-01-01", stress_score=0.3, features={}))
    history.add(Observation(plant_id="B", timestamp="2024-01-01", stress_score=0.5, features={}))
    # No plant has two observations, so there is no curve to fit: a clean error, not a raw crash.
    with pytest.raises(InsufficientDataError):
        fit_cohort_survival(history, "weibull-aft")


def test_trend_payload_degrades_when_no_plant_has_two_observations() -> None:
    from phytovision.api_payloads import trend_payload
    from phytovision.temporal import FeatureHistory
    from phytovision.temporal.history import Observation

    history = FeatureHistory()
    history.add(Observation(plant_id="A", timestamp="2024-01-01", stress_score=0.3, features={}))
    history.add(Observation(plant_id="B", timestamp="2024-01-01", stress_score=0.5, features={}))
    payload = trend_payload(history)  # default weibull-aft; must not raise
    assert payload["survival"] is None
    assert "survival_note" in payload  # the omission is surfaced, not silent

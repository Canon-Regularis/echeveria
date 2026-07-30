"""The forecaster benchmark harness and its CLI."""

from __future__ import annotations

import csv

import pytest

from phytovision.cli import main
from phytovision.evaluation.benchmark import benchmark_forecasters
from phytovision.models.forecasting.base import Prediction, SeriesForecaster
from phytovision.registries import FORECASTERS
from phytovision.simulation import DryDownParams, cohort_history, simulate_cohort, write_manifest


class _MissingExtra(SeriesForecaster):
    name = "benchmark-missing-extra"

    def _predict(self, scores, steps):  # type: ignore[no-untyped-def]
        raise ImportError("needs an extra that is not installed")


class _AlwaysFallback(SeriesForecaster):
    name = "benchmark-fallback"

    def _predict(self, scores, steps):  # type: ignore[no-untyped-def]
        raise ValueError("cannot fit; degrade to the linear interval")


class _Constant(SeriesForecaster):
    name = "benchmark-constant"

    def _predict(self, scores, steps):  # type: ignore[no-untyped-def]
        last = scores[-1]
        return Prediction(
            {h: last for h in steps},
            {h: max(0.0, last - 0.1) for h in steps},
            {h: min(1.0, last + 0.1) for h in steps},
        )


# Register the throwaway forecasters once, so the skip path and the ranking test have concrete
# forecasters to run. Registration persists for the session, so guard against a re-import.
if "benchmark-missing-extra" not in FORECASTERS:
    FORECASTERS.register("benchmark-missing-extra")(_MissingExtra)
if "benchmark-constant" not in FORECASTERS:
    FORECASTERS.register("benchmark-constant")(_Constant)
if "benchmark-fallback" not in FORECASTERS:
    FORECASTERS.register("benchmark-fallback")(_AlwaysFallback)


def test_benchmark_ranks_forecasters_by_crps() -> None:
    history = cohort_history(simulate_cohort(6, DryDownParams(n_steps=12), seed=1))
    result = benchmark_forecasters(history, ["linear-trend"], horizons=(1, 3), min_train=4)
    assert result.horizons() == [1, 3]
    for horizon in result.horizons():
        row = result.for_horizon(horizon)
        assert len(row) == 1
        assert row[0].name == "linear-trend"
        assert row[0].n > 0
        assert 0.0 <= row[0].coverage <= 1.0


def test_crps_ci_brackets_the_point_estimate_and_is_reproducible() -> None:
    history = cohort_history(simulate_cohort(12, DryDownParams(n_steps=14), seed=3))
    first = benchmark_forecasters(history, ["linear-trend"], horizons=(1,), min_train=4, seed=0)
    again = benchmark_forecasters(history, ["linear-trend"], horizons=(1,), min_train=4, seed=0)
    score = first.for_horizon(1)[0]
    low, high = score.crps_ci95
    assert low <= score.crps <= high
    assert first.for_horizon(1)[0].crps_ci95 == again.for_horizon(1)[0].crps_ci95  # seeded


def test_clustered_ci_is_wider_than_the_naive_per_observation_ci() -> None:
    # Observations within a plant are correlated. Pooling them as independent (the per-observation
    # normal approximation) reports a CI that is too narrow; the per-plant cluster bootstrap widens
    # it to reflect how few independent units there really are.
    import numpy as np

    from phytovision.evaluation._aggregate import mean_ci95
    from phytovision.evaluation.benchmark import _clustered_ci95

    rng = np.random.default_rng(0)
    groups: list[int] = []
    samples: list[float] = []
    for plant in range(10):
        offset = float(rng.normal(0.0, 0.3))  # a per-plant level: within-plant correlation
        for _ in range(20):
            groups.append(plant)
            samples.append(0.5 + offset + float(rng.normal(0.0, 0.02)))
    naive = mean_ci95(samples)
    clustered = _clustered_ci95(samples, groups, seed=0)
    assert (clustered[1] - clustered[0]) > 2 * (naive[1] - naive[0])
    assert _clustered_ci95(samples, [0] * len(samples), seed=0) == pytest.approx(
        (float(np.mean(samples)), float(np.mean(samples)))
    )  # a single plant carries no between-plant spread


def test_benchmark_table_is_sorted_within_each_horizon() -> None:
    history = cohort_history(simulate_cohort(6, DryDownParams(n_steps=12), seed=2))
    result = benchmark_forecasters(
        history, ["linear-trend", "benchmark-constant"], horizons=(1,), min_train=4
    )
    rows = [r for r in result.table() if r["horizon"] == 1]
    crps = [r["crps"] for r in rows]
    assert crps == sorted(crps)  # best CRPS first


def test_benchmark_skips_a_forecaster_missing_its_extra() -> None:
    history = cohort_history(simulate_cohort(5, DryDownParams(n_steps=10), seed=3))
    result = benchmark_forecasters(
        history, ["linear-trend", "benchmark-missing-extra"], horizons=(1,), min_train=4
    )
    assert result.skipped == ("benchmark-missing-extra",)
    assert {score.name for score in result.scores} == {"linear-trend"}


def test_benchmark_surfaces_a_forecaster_that_fell_back() -> None:
    # A forecaster that could not fit on every origin is still scored (linear fallback numbers), but
    # its row must be flagged so those numbers are not read as pure model output.
    history = cohort_history(simulate_cohort(5, DryDownParams(n_steps=10), seed=3))
    result = benchmark_forecasters(
        history, ["linear-trend", "benchmark-fallback"], horizons=(1,), min_train=4
    )
    assert result.fallbacks == ("benchmark-fallback",)
    assert "benchmark-fallback" in {score.name for score in result.scores}  # still scored
    assert "linear-trend" not in result.fallbacks  # a genuine fit is not flagged


def test_cli_benchmark_ranks_over_a_cohort(tmp_path, capsys) -> None:
    cohort = simulate_cohort(6, DryDownParams(n_steps=14), seed=4)
    manifest = write_manifest(cohort, tmp_path / "cohort.csv")
    out = tmp_path / "table.csv"
    argv = ["benchmark", str(manifest), "--horizons", "1,3", "--forecasters", "linear-trend"]
    assert main([*argv, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "synthetic" in printed.lower()
    rows = list(csv.DictReader(out.open()))
    assert {row["forecaster"] for row in rows} == {"linear-trend"}
    assert {int(row["horizon"]) for row in rows} == {1, 3}


def test_cli_benchmark_rejects_an_unknown_forecaster(tmp_path, capsys) -> None:
    cohort = simulate_cohort(3, DryDownParams(n_steps=8), seed=5)
    manifest = write_manifest(cohort, tmp_path / "c.csv")
    rc = main(["benchmark", str(manifest), "--forecasters", "nope"])
    assert rc == 2
    assert "unknown forecaster" in capsys.readouterr().err


def test_cli_benchmark_missing_score_column_errors(tmp_path, capsys) -> None:
    manifest = tmp_path / "images.csv"
    manifest.write_text("image_path,plant_id,timestamp\na.png,p1,2026-03-01\n")
    rc = main(["benchmark", str(manifest)])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_benchmark_mlflow_without_the_extra_reports_a_clean_error(tmp_path, capsys) -> None:
    if _mlflow_installed():
        pytest.skip("mlflow is installed, so the missing-extra path cannot be exercised")
    cohort = simulate_cohort(4, DryDownParams(n_steps=10), seed=6)
    manifest = write_manifest(cohort, tmp_path / "c.csv")
    rc = main(["benchmark", str(manifest), "--forecasters", "linear-trend", "--mlflow"])
    assert rc == 2
    assert "tracking extra" in capsys.readouterr().err


def test_cli_benchmark_mlflow_runtime_failure_is_a_clean_error(
    tmp_path, capsys, monkeypatch
) -> None:
    # A tracking-store failure (read-only dir, unreachable URI) after the benchmark already ran must
    # be a clean error, not a traceback that discards the ranked table. Patch the logger to raise a
    # non-ImportError so this holds whether or not mlflow is installed.
    import phytovision.tracking as tracking

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("tracking store is read-only")

    monkeypatch.setattr(tracking, "log_benchmark", _boom)
    cohort = simulate_cohort(4, DryDownParams(n_steps=10), seed=7)
    manifest = write_manifest(cohort, tmp_path / "c.csv")
    rc = main(["benchmark", str(manifest), "--forecasters", "linear-trend", "--mlflow"])
    assert rc == 2
    assert "MLflow logging failed" in capsys.readouterr().err


def _mlflow_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("mlflow") is not None


def test_selected_forecaster_names_are_deduplicated() -> None:
    # A repeated name ran the whole expanding-window comparison twice and ranked the same forecaster
    # twice in the table, for no additional information.
    from phytovision.cli.benchmark import _selected_names

    assert _selected_names("arima,arima") == ["arima"]
    assert _selected_names("arima, linear-trend ,arima") == ["arima", "linear-trend"]
    assert _selected_names("") is None


def test_cli_benchmark_bad_out_is_a_clean_error(tmp_path, capsys) -> None:
    # A --out whose parent directory does not exist must be a clean error, not an uncaught OSError
    # traceback after the ranked table was already computed.
    manifest = write_manifest(
        simulate_cohort(5, DryDownParams(n_steps=12), seed=8), tmp_path / "c.csv"
    )
    rc = main(
        [
            "benchmark",
            str(manifest),
            "--forecasters",
            "linear-trend",
            "--out",
            str(tmp_path / "missing_dir" / "table.csv"),
        ]
    )
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_benchmark_no_forecasts_scored_is_a_clean_error(tmp_path, capsys) -> None:
    # A --min-train larger than every plant's series yields no expanding-window origins, so nothing
    # is scored; the command must say so and exit 2 rather than print an empty table and exit 0.
    manifest = write_manifest(
        simulate_cohort(4, DryDownParams(n_steps=8), seed=9), tmp_path / "c.csv"
    )
    rc = main(["benchmark", str(manifest), "--forecasters", "linear-trend", "--min-train", "100"])
    assert rc == 2
    assert "no forecasts were scored" in capsys.readouterr().err

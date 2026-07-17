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


def _mlflow_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("mlflow") is not None

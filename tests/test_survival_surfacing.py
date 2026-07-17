"""Survival on the surfaces: the /trend block, the dashboard helpers, and the phenotype columns."""

from __future__ import annotations

import csv
import importlib.util

import numpy as np
import pytest

from phytovision.cli import main
from phytovision.cli.phenotype import survival_row
from phytovision.dashboard import plant_survival_metrics, survival_curve_points
from phytovision.models.survival import PlantSurvival, SurvivalCurve, SurvivalFit
from phytovision.simulation import DryDownParams, cohort_history, simulate_cohort

_HAS_LIFELINES = importlib.util.find_spec("lifelines") is not None
_needs_lifelines = pytest.mark.skipif(
    not _HAS_LIFELINES, reason="needs the stats extra (lifelines)"
)


def _history(n: int = 8):
    params = DryDownParams(n_steps=12, base_decline_rate=0.16, decline_rate_spread=0.5)
    return cohort_history(simulate_cohort(n, params, seed=2))


def _save_image(path, image) -> None:
    from PIL import Image as PILImage

    PILImage.fromarray((image * 255).astype(np.uint8)).save(path)


def _hand_fit() -> SurvivalFit:
    curve = SurvivalCurve((1.0, 2.0, 3.0), (1.0, 0.7, 0.4), (0.9, 0.6, 0.3), (1.0, 0.8, 0.5))
    per_plant = {"p1": PlantSurvival("p1", 2.5, 2.0, 3.2, "weibull-aft", None)}
    return SurvivalFit("weibull-aft", curve, 2.0, (1.0, 3.0), 0.82, per_plant, "note")


# --- dashboard helpers and the phenotype column helper (base-deps, always run) ---


def test_survival_curve_points_are_aligned_and_non_increasing() -> None:
    times, survival, lower, upper = survival_curve_points(_hand_fit())
    assert len(times) == len(survival) == len(lower) == len(upper) == 3
    assert all(a >= b for a, b in zip(survival, survival[1:], strict=False))


def test_plant_survival_metrics_reads_a_plant_and_a_missing_plant() -> None:
    fit = _hand_fit()
    metrics = plant_survival_metrics(fit, "p1")
    assert metrics == {"median": 2.5, "lower": 2.0, "upper": 3.2, "basis": "weibull-aft"}
    assert plant_survival_metrics(fit, "absent")["basis"] == "unavailable"


def test_survival_row_blank_when_unavailable() -> None:
    row = survival_row(None, "p1")
    assert row["survival_basis"] == "unavailable-stats-extra"
    assert row["median_time_to_wilt"] == "" and row["time_to_wilt_lo"] == ""


def test_survival_row_reads_a_finite_plant() -> None:
    row = survival_row(_hand_fit(), "p1")
    assert row["survival_basis"] == "weibull-aft"
    assert row["median_time_to_wilt"] == 2.5


def test_survival_row_names_the_dropped_plant_honestly() -> None:
    # The fit ran (not None) but this plant was dropped for too few observations, so the missing
    # extra must not be blamed.
    row = survival_row(_hand_fit(), "p_absent")
    assert row["survival_basis"] == "insufficient-observations"
    assert row["median_time_to_wilt"] == ""


# --- the API /trend survival block ---


@_needs_lifelines
def test_trend_payload_carries_a_survival_block() -> None:
    from phytovision.api_payloads import trend_payload

    payload = trend_payload(_history(), None, "weibull-aft")
    summary = payload["survival"]
    assert isinstance(summary, dict)
    assert {"model", "cohort_median", "cohort_median_ci", "concordance_index", "curve"} <= set(
        summary
    )
    plant = next(iter(payload["plants"].values()))
    assert set(plant["survival"]) == {"basis", "median", "lower", "upper"}  # type: ignore[index]
    assert "survival estimate" in payload["disclaimer"]  # type: ignore[operator]


def test_trend_payload_degrades_without_the_stats_extra(monkeypatch) -> None:
    import phytovision.api_payloads as payloads

    def _no_extra(*_args: object, **_kwargs: object) -> object:
        raise ImportError("survival needs the stats extra")

    monkeypatch.setattr(payloads, "fit_cohort_survival", _no_extra)
    payload = payloads.trend_payload(_history(), None, "weibull-aft")
    assert payload["survival"] is None
    assert "survival_note" in payload
    assert payload["plants"]  # the forecast is still delivered


# --- the phenotype CSV columns ---


@_needs_lifelines
def test_phenotype_writes_survival_columns(tmp_path, healthy_image, stressed_image) -> None:
    _save_image(tmp_path / "p1_t1.png", healthy_image)
    _save_image(tmp_path / "p1_t2.png", stressed_image)
    _save_image(tmp_path / "p2_t1.png", healthy_image)
    _save_image(tmp_path / "p2_t2.png", stressed_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "image_path,plant_id,timestamp\n"
        "p1_t1.png,p1,2026-03-01\n"
        "p1_t2.png,p1,2026-03-02\n"
        "p2_t1.png,p2,2026-03-01\n"
        "p2_t2.png,p2,2026-03-02\n"
    )
    out = tmp_path / "traj.csv"
    argv = ["phenotype", str(manifest), "--out", str(out), "--survival-model", "weibull-aft"]
    assert main(argv) == 0
    row = next(csv.DictReader(out.open()))
    assert {"median_time_to_wilt", "time_to_wilt_lo", "time_to_wilt_hi", "survival_basis"} <= set(
        row
    )
    assert row["survival_basis"] in {"weibull-aft", "cohort-km"}


def test_phenotype_degrades_without_the_stats_extra(
    tmp_path, healthy_image, stressed_image, monkeypatch, capsys
) -> None:
    import phytovision.cli.phenotype as phenotype

    def _no_extra(*_args: object, **_kwargs: object) -> object:
        raise ImportError("survival needs the stats extra")

    monkeypatch.setattr(phenotype, "fit_cohort_survival", _no_extra)
    _save_image(tmp_path / "a.png", healthy_image)
    _save_image(tmp_path / "b.png", stressed_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,plant_id,timestamp\na.png,p1,2026-03-01\nb.png,p1,2026-03-02\n")
    out = tmp_path / "traj.csv"
    assert main(["phenotype", str(manifest), "--out", str(out)]) == 0
    assert "survival omitted" in capsys.readouterr().out
    row = next(csv.DictReader(out.open()))
    assert row["survival_basis"] == "unavailable-stats-extra"
    assert row["median_time_to_wilt"] == ""

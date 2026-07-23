"""CLI behaviour: success, error handling, JSON output."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from phytovision.cli import _parse_horizons, main
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.persistence import load_saved
from phytovision.models.stress.ensemble import EnsembleStressModel


def test_cli_analyze_succeeds(image_path) -> None:
    assert main(["analyze", str(image_path)]) == 0


def test_cli_json_output_is_valid(image_path, capsys) -> None:
    rc = main(["analyze", str(image_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["stress"]) == {"score", "confidence", "label", "model"}


def test_cli_missing_file_reports_error_to_stderr(capsys) -> None:
    rc = main(["analyze", "does-not-exist.png"])
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert captured.out == ""


def test_cli_model_selection(image_path) -> None:
    assert main(["analyze", str(image_path), "--model", "heuristic"]) == 0


def test_cli_unbuildable_model_reports_clean_error(image_path, capsys) -> None:
    # gradient-boosted needs training data, so building it by name must fail cleanly:
    # a clean "error: ..." with exit code 2, not an uncaught traceback.
    rc = main(["analyze", str(image_path), "--model", "gradient-boosted"])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_analyze_features_flag(image_path, capsys) -> None:
    rc = main(["analyze", str(image_path), "--json", "--features"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "features" in payload
    assert "colour.gcc_mean" in payload["features"]


def test_cli_save_overlay(image_path, tmp_path) -> None:
    out = tmp_path / "overlay.png"
    assert main(["analyze", str(image_path), "--save-overlay", str(out)]) == 0
    assert out.exists()


def test_cli_save_saliency(image_path, tmp_path, capsys) -> None:
    out = tmp_path / "saliency.png"
    assert main(["analyze", str(image_path), "--save-saliency", str(out)]) == 0
    assert out.exists()
    assert "Saliency written" in capsys.readouterr().out


def test_cli_save_with_unwritable_extension_is_a_clean_error(image_path, capsys) -> None:
    # A save path with no recognised image extension used to reach PIL.save() and raise a raw
    # ValueError traceback; it is now rejected up front with a clean error and exit code 2.
    rc = main(["analyze", str(image_path), "--save-overlay", "out.txt"])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_batch_writes_csv(dataset_dir, tmp_path) -> None:
    out = tmp_path / "features.csv"
    assert main(["batch", str(dataset_dir), "--out", str(out)]) == 0
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    assert {r["label"] for r in rows} == {"healthy", "wilted"}
    assert "colour.gcc_mean" in rows[0]


def test_cli_batch_writes_json(dataset_dir, tmp_path) -> None:
    out = tmp_path / "features.json"
    assert main(["batch", str(dataset_dir), "--out", str(out)]) == 0
    records = json.loads(out.read_text())
    assert len(records) == 2
    assert "colour.gcc_mean" in records[0]


def test_cli_batch_empty_dir_errors(tmp_path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = main(["batch", str(empty), "--out", str(tmp_path / "x.csv")])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_batch_bad_out_reports_clean_error(dataset_dir, tmp_path, capsys) -> None:
    # A bad --out (a missing parent directory) is a clean error with exit code 2, not a traceback.
    bad_out = tmp_path / "missing_dir" / "out.csv"
    rc = main(["batch", str(dataset_dir), "--out", str(bad_out)])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def _save_image(path, image) -> None:
    from PIL import Image as PILImage

    PILImage.fromarray((image * 255).astype(np.uint8)).save(path)


def test_cli_validate_reports_calibration_and_regression(
    tmp_path, capsys, healthy_image, stressed_image
) -> None:
    _save_image(tmp_path / "a.png", healthy_image)
    _save_image(tmp_path / "b.png", stressed_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,label,target\na.png,healthy,0.1\nb.png,wilted,0.8\n")
    assert main(["validate", str(manifest)]) == 0
    out = capsys.readouterr().out
    assert "reliability" in out
    assert "RMSE" in out


def test_cli_validate_without_targets_skips_regression(tmp_path, capsys, healthy_image) -> None:
    _save_image(tmp_path / "a.png", healthy_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,label\na.png,healthy\n")
    assert main(["validate", str(manifest)]) == 0
    assert "regression is skipped" in capsys.readouterr().out


def test_cli_validate_without_labels_skips_the_reliability_curve(
    tmp_path, capsys, healthy_image
) -> None:
    # An unlabelled manifest must not fabricate a calibration report: an absent label reads as a
    # true stressed event, which used to assert every image was stressed with a huge Brier score.
    _save_image(tmp_path / "a.png", healthy_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,target\na.png,0.1\n")
    assert main(["validate", str(manifest)]) == 0
    out = capsys.readouterr().out
    assert "reliability curve is skipped" in out
    assert "Brier score" not in out


def test_cli_simulate_rejects_a_non_positive_step_count(tmp_path, capsys) -> None:
    # --steps 0 used to write one observation per plant and report a step count that never happened.
    rc = main(["simulate", "--out", str(tmp_path / "c.csv"), "--plants", "3", "--steps", "0"])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_phenotype_writes_a_trajectory_table(tmp_path, healthy_image, stressed_image) -> None:
    _save_image(tmp_path / "p1_t1.png", healthy_image)
    _save_image(tmp_path / "p1_t2.png", stressed_image)
    _save_image(tmp_path / "p2_t1.png", healthy_image)
    _save_image(tmp_path / "p2_t2.png", healthy_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "image_path,plant_id,timestamp\n"
        "p1_t1.png,p1,2026-03-01\n"
        "p1_t2.png,p1,2026-03-02\n"
        "p2_t1.png,p2,2026-03-01\n"
        "p2_t2.png,p2,2026-03-02\n"
    )
    out = tmp_path / "traj.csv"
    assert main(["phenotype", str(manifest), "--out", str(out), "--horizons", "1,3"]) == 0

    rows = list(csv.DictReader(out.open()))
    assert {row["plant_id"] for row in rows} == {"p1", "p2"}
    p1 = next(row for row in rows if row["plant_id"] == "p1")
    assert p1["latest_stage"] in {"well-watered", "early-stress", "moderate", "severe"}
    assert {"forecast_h1", "forecast_h3", "forecast_confidence"} <= set(p1)
    assert p1["trend_direction"] == "rising"  # healthy then stressed


def test_cli_phenotype_emits_prediction_intervals(tmp_path, healthy_image, stressed_image) -> None:
    _save_image(tmp_path / "p1_t1.png", healthy_image)
    _save_image(tmp_path / "p1_t2.png", stressed_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "image_path,plant_id,timestamp\np1_t1.png,p1,2026-03-01\np1_t2.png,p1,2026-03-02\n"
    )
    out = tmp_path / "traj.csv"
    argv = ["phenotype", str(manifest), "--out", str(out), "--horizons", "1,3"]
    assert main([*argv, "--forecaster", "linear-trend"]) == 0
    row = next(csv.DictReader(out.open()))
    assert row["forecast_method"] == "linear-trend"
    for horizon in ("1", "3"):
        lo = float(row[f"forecast_h{horizon}_lo"])
        hi = float(row[f"forecast_h{horizon}_hi"])
        mean = float(row[f"forecast_h{horizon}"])
        assert lo <= mean <= hi


def test_cli_phenotype_survives_single_observation_cohort(tmp_path, healthy_image) -> None:
    _save_image(tmp_path / "p1.png", healthy_image)
    _save_image(tmp_path / "p2.png", healthy_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "image_path,plant_id,timestamp\np1.png,p1,2026-03-01\np2.png,p2,2026-03-01\n"
    )
    out = tmp_path / "traj.csv"
    # Every plant has a single observation, so the survival fit raises InsufficientDataError;
    # phenotype must omit survival and still write the table rather than crash.
    assert main(["phenotype", str(manifest), "--out", str(out)]) == 0
    assert out.exists()


def test_cli_benchmark_rejects_min_train_below_two() -> None:
    # The guard fires before the manifest is read, so a clean exit 2 results, not a traceback.
    assert main(["benchmark", "unused.csv", "--min-train", "1"]) == 2


def test_cli_phenotype_fails_cleanly_on_missing_forecaster_extra(
    tmp_path, healthy_image, monkeypatch
) -> None:
    import phytovision.cli.phenotype as phenotype

    def _missing_extra(*args, **kwargs):
        raise ImportError('the forecaster needs the stats extra: pip install -e ".[stats]"')

    monkeypatch.setattr(phenotype, "plant_forecasts", _missing_extra)
    _save_image(tmp_path / "p1_t1.png", healthy_image)
    _save_image(tmp_path / "p1_t2.png", healthy_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "image_path,plant_id,timestamp\np1_t1.png,p1,2026-03-01\np1_t2.png,p1,2026-03-02\n"
    )
    # A forecaster whose optional extra is absent must fail cleanly (exit 2), not crash the command.
    assert main(["phenotype", str(manifest), "--out", str(tmp_path / "traj.csv")]) == 2


def test_cli_phenotype_json_uses_default_horizons(tmp_path, healthy_image, stressed_image) -> None:
    _save_image(tmp_path / "a.png", healthy_image)
    _save_image(tmp_path / "b.png", stressed_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,plant_id,timestamp\na.png,p1,2026-03-01\nb.png,p1,2026-03-02\n")
    out = tmp_path / "traj.json"
    assert main(["phenotype", str(manifest), "--out", str(out)]) == 0
    records = json.loads(out.read_text())
    assert len(records) == 1
    assert "forecast_h7" in records[0]  # default horizons are 1,3,7


def test_parse_horizons_is_positive_deduped_and_sorted() -> None:
    from phytovision.temporal import DEFAULT_HORIZONS

    assert _parse_horizons("1,3,7") == (1, 3, 7)
    assert _parse_horizons("  2 , 5 ") == (2, 5)  # tolerates whitespace
    assert _parse_horizons("-1,2,3") == (2, 3)  # non-positive dropped
    assert _parse_horizons("3,1,1,2") == (1, 2, 3)  # deduped and sorted (no duplicate columns)
    assert _parse_horizons("") == DEFAULT_HORIZONS  # empty -> defaults
    assert _parse_horizons("abc") == DEFAULT_HORIZONS  # unparseable -> defaults
    assert _parse_horizons("0,-5") == DEFAULT_HORIZONS  # nothing positive -> defaults


def test_cli_phenotype_no_tagged_rows_errors(tmp_path, healthy_image, capsys) -> None:
    _save_image(tmp_path / "a.png", healthy_image)
    manifest = tmp_path / "m.csv"
    manifest.write_text("image_path,plant_id,timestamp\na.png,,\n")  # no plant_id/timestamp
    rc = main(["phenotype", str(manifest), "--out", str(tmp_path / "x.csv")])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_config_toml_runs(image_path, tmp_path) -> None:
    cfg = tmp_path / "pipeline.toml"
    cfg.write_text(
        '[preprocessor]\nname = "resize-normalize"\n[preprocessor.params]\nmax_size = 256\n'
    )
    assert main(["analyze", str(image_path), "--config", str(cfg)]) == 0


def test_cli_config_unknown_component_errors(image_path, tmp_path, capsys) -> None:
    cfg = tmp_path / "pipeline.json"
    cfg.write_text(json.dumps({"model": "does-not-exist"}))
    rc = main(["analyze", str(image_path), "--config", str(cfg)])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_train_then_use_via_model_path(training_dir, tmp_path) -> None:
    pytest.importorskip("sklearn")
    model_path = tmp_path / "model.joblib"
    assert main(["train", str(training_dir), "--out", str(model_path)]) == 0
    assert model_path.exists()

    image = next((training_dir / "healthy").glob("*.png"))
    assert main(["analyze", str(image), "--model-path", str(model_path)]) == 0


def test_cli_train_seed_is_recorded_in_the_manifest(training_dir, tmp_path) -> None:
    pytest.importorskip("sklearn")
    from phytovision.models.persistence import read_envelope

    model_path = tmp_path / "seeded.joblib"
    assert main(["train", str(training_dir), "--out", str(model_path), "--seed", "0"]) == 0
    assert read_envelope(model_path)["manifest"]["seed"] == 0


def test_cli_train_single_class_errors(tmp_path, capsys, healthy_image) -> None:
    from PIL import Image as PILImage

    class_dir = tmp_path / "one" / "healthy"
    class_dir.mkdir(parents=True)
    PILImage.fromarray((healthy_image * 255).astype(np.uint8)).save(class_dir / "a.png")

    rc = main(["train", str(tmp_path / "one"), "--out", str(tmp_path / "model.joblib")])
    assert rc == 2
    assert "two classes" in capsys.readouterr().err


def test_cli_evaluate_runs(dataset_dir, capsys) -> None:
    rc = main(["evaluate", str(dataset_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "confusion" in out


def test_cli_serve_without_uvicorn_reports_clean_error(monkeypatch, capsys) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", None)  # force `import uvicorn` to raise ImportError
    rc = main(["serve"])
    assert rc == 2
    assert "api" in capsys.readouterr().err


def test_cli_serve_bad_model_path_reports_clean_error(monkeypatch, tmp_path, capsys) -> None:
    import sys
    import types

    monkeypatch.setitem(sys.modules, "uvicorn", types.ModuleType("uvicorn"))  # importable stub
    rc = main(["serve", "--model-path", str(tmp_path / "missing.joblib")])
    assert rc == 2
    assert capsys.readouterr().err.startswith("error:")


def test_cli_dashboard_without_streamlit_reports_clean_error(monkeypatch, capsys) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "streamlit", None)  # force `import streamlit` to raise
    rc = main(["dashboard"])
    assert rc == 2
    assert "dashboard" in capsys.readouterr().err


def test_cli_dashboard_bad_model_path_reports_clean_error(monkeypatch, tmp_path, capsys) -> None:
    import sys
    import types

    monkeypatch.setitem(sys.modules, "streamlit", types.ModuleType("streamlit"))  # importable stub
    rc = main(["dashboard", "--model-path", str(tmp_path / "missing.joblib")])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "missing.joblib" in err  # confirms we reached model-path validation, not an earlier exit


def test_cli_train_ensemble_saves_and_loads(training_dir, image_path, tmp_path, capsys) -> None:
    pytest.importorskip("sklearn")
    out = tmp_path / "ensemble.joblib"
    assert main(["train", str(training_dir), "--model", "ensemble", "--out", str(out)]) == 0
    assert isinstance(load_saved(out), EnsembleStressModel)

    capsys.readouterr()
    assert main(["analyze", str(image_path), "--model-path", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stress"]["model"] == "ensemble-v1"


def test_cli_train_calibrate_then_analyze_conformal(
    training_dir, image_path, tmp_path, capsys
) -> None:
    pytest.importorskip("sklearn")
    out = tmp_path / "calibrated.joblib"
    assert (
        main(
            ["train", str(training_dir), "--calibrate", "0.3", "--alpha", "0.1", "--out", str(out)]
        )
        == 0
    )
    assert isinstance(load_saved(out), SplitConformalClassifier)

    capsys.readouterr()
    assert main(["analyze", str(image_path), "--model-path", str(out), "--conformal"]) == 0
    assert "Conformal set" in capsys.readouterr().out

    assert (
        main(["analyze", str(image_path), "--model-path", str(out), "--conformal", "--json"]) == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert "labels" in payload["conformal"] and "alpha" in payload["conformal"]


def test_cli_analyze_conformal_without_calibrated_model_errors(image_path, capsys) -> None:
    assert main(["analyze", str(image_path), "--conformal"]) == 2
    assert "train --calibrate" in capsys.readouterr().err


def test_cli_analyze_omits_conformal_without_the_flag(
    training_dir, image_path, tmp_path, capsys
) -> None:
    pytest.importorskip("sklearn")
    out = tmp_path / "calibrated.joblib"
    assert (
        main(
            ["train", str(training_dir), "--calibrate", "0.3", "--alpha", "0.1", "--out", str(out)]
        )
        == 0
    )
    capsys.readouterr()
    # Loading the calibrated model without --conformal uses its classifier but must not inject a
    # conformal block into the output the caller never asked for.
    assert main(["analyze", str(image_path), "--model-path", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "conformal" not in payload


def test_cli_train_calibrate_rejects_bad_fraction(training_dir, tmp_path, capsys) -> None:
    out = tmp_path / "x.joblib"
    assert main(["train", str(training_dir), "--calibrate", "1.5", "--out", str(out)]) == 2
    assert "fraction in (0, 1)" in capsys.readouterr().err


def test_cli_evaluate_cv_uses_the_selected_model(training_dir, capsys) -> None:
    pytest.importorskip("sklearn")
    assert main(["evaluate", str(training_dir), "--model", "ensemble", "--cv", "3"]) == 0
    assert "cross-validation (ensemble)" in capsys.readouterr().out


def test_cli_evaluate_importance(training_dir, capsys) -> None:
    pytest.importorskip("sklearn")
    assert main(["evaluate", str(training_dir), "--importance"]) == 0
    assert "permutation feature importance" in capsys.readouterr().out


def test_cli_evaluate_cv_explicit_gradient_boosted(training_dir, capsys) -> None:
    # --model gradient-boosted must not crash while building the feature-extraction pipeline: cv
    # retrains a model per fold, so the extraction model stays a buildable default.
    pytest.importorskip("sklearn")
    assert main(["evaluate", str(training_dir), "--model", "gradient-boosted", "--cv", "3"]) == 0
    assert "cross-validation (gradient-boosted)" in capsys.readouterr().out


def test_cli_evaluate_transfer_uses_the_selected_model(transfer_dirs, capsys) -> None:
    pytest.importorskip("sklearn")
    argv = [
        "evaluate",
        str(transfer_dirs[0]),
        str(transfer_dirs[1]),
        "--model",
        "ensemble",
        "--transfer",
    ]
    assert main(argv) == 0
    assert "leave-one-dataset-out (ensemble" in capsys.readouterr().out


def test_cli_analyze_with_shap_explainer(training_dir, image_path, tmp_path, capsys) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("shap")
    model_path = tmp_path / "gb.joblib"
    assert main(["train", str(training_dir), "--out", str(model_path)]) == 0

    capsys.readouterr()
    argv = [
        "analyze",
        str(image_path),
        "--model-path",
        str(model_path),
        "--explainer",
        "shap",
        "--json",
    ]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["explanation_method"] == "shap"
    assert "additivity_error" in payload


def test_cli_shap_explainer_degrades_without_a_shap_model(image_path, capsys) -> None:
    # The heuristic is not SHAP-attributable, so --explainer shap must degrade, not crash.
    assert main(["analyze", str(image_path), "--explainer", "shap", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["explanation_method"] == "shap-unavailable"


def test_cli_analyze_timing(image_path, capsys) -> None:
    assert main(["analyze", str(image_path), "--timing"]) == 0
    assert "Timing (ms):" in capsys.readouterr().out


def test_cli_analyze_disease_head(image_path, capsys) -> None:
    assert main(["analyze", str(image_path), "--disease", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # The head must attach AND emit its two-class distribution, not just an empty slot.
    assert set(payload["head_outputs"]["disease"]) == {"healthy", "lesion-like"}


def test_cli_analyze_drought_stage_head(image_path, capsys) -> None:
    assert main(["analyze", str(image_path), "--drought-stage", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    stage = payload["head_outputs"]["drought_stage"]
    assert stage["stage"] in {"well-watered", "early-stress", "moderate", "severe"}
    assert set(stage["markers"]) == {"pigment", "turgor_loss", "necrosis"}


def test_cli_analyze_counterfactual(image_path, capsys) -> None:
    assert main(["analyze", str(image_path), "--counterfactual", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "counterfactuals" in payload  # a list (possibly empty for a clearly-healthy plant)


def test_cli_strict_schema_flags_extractor_drift(
    training_dir, image_path, tmp_path, capsys
) -> None:
    pytest.importorskip("sklearn")
    model_path = tmp_path / "gb.joblib"
    assert main(["train", str(training_dir), "--out", str(model_path)]) == 0  # full extractor stack

    # Fewer extractors produce fewer features than the model was trained on.
    cfg = tmp_path / "fewer.json"
    cfg.write_text(json.dumps({"feature_extractors": ["geometry", "colour"]}))
    base = ["analyze", str(image_path), "--model-path", str(model_path), "--config", str(cfg)]

    assert main([*base, "--strict-schema"]) == 2
    assert "schema mismatch" in capsys.readouterr().err
    assert main(base) == 0  # tolerant mode warns but still analyzes

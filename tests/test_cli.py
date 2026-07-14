"""CLI behaviour: success, error handling, JSON output."""

from __future__ import annotations

import csv
import json

from phytovision.cli import main


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

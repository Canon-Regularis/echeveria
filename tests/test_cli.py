"""CLI behaviour: success, error handling, JSON output."""

from __future__ import annotations

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

"""Leave-one-dataset-out transfer evaluation (F14). Requires the ``ml`` extra."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from phytovision.analysis import AnalysisRow  # noqa: E402
from phytovision.cli import main  # noqa: E402
from phytovision.evaluation.cross_dataset import (  # noqa: E402
    TransferMatrix,
    leave_one_dataset_out,
)
from phytovision.evaluation.metrics import BinaryMetrics  # noqa: E402
from phytovision.exceptions import ConfigError  # noqa: E402


def _row(label: str, source: str | None, seed: int) -> AnalysisRow:
    rng = np.random.default_rng(seed)
    if label == "healthy":
        features = {"a": float(rng.normal(0.40, 0.02)), "b": float(abs(rng.normal(0.03, 0.02)))}
    else:
        features = {"a": float(rng.normal(0.30, 0.02)), "b": float(abs(rng.normal(0.40, 0.05)))}
    return AnalysisRow("x", label, None, source, 0.0, 0.0, "healthy", "m", features)


def _dataset(source: str, labels: list[str], base_seed: int) -> list[AnalysisRow]:
    return [_row(label, source, base_seed + i) for i, label in enumerate(labels)]


def test_matrix_has_one_entry_per_dataset() -> None:
    rows = _dataset("d1", ["healthy", "wilted"] * 5, 0) + _dataset(
        "d2", ["healthy", "wilted"] * 5, 100
    )
    result = leave_one_dataset_out(rows)
    assert isinstance(result, TransferMatrix)
    assert set(result.datasets) == {"d1", "d2"}
    assert 0.0 <= result.mean_accuracy <= 1.0
    assert isinstance(result.metrics_for("d1"), BinaryMetrics)


def test_needs_at_least_two_datasets() -> None:
    rows = _dataset("only", ["healthy", "wilted"] * 5, 0)
    with pytest.raises(ConfigError, match="at least two datasets"):
        leave_one_dataset_out(rows)


def test_skips_a_hold_out_that_leaves_single_class_training() -> None:
    # d1 and d2 are healthy-only; d3 has both. Holding out d3 leaves training all-healthy, so it is
    # skipped. Holding out d1 or d2 keeps both classes in training via d3.
    rows = (
        _dataset("d1", ["healthy"] * 4, 0)
        + _dataset("d2", ["healthy"] * 4, 50)
        + _dataset("d3", ["healthy", "wilted"] * 4, 100)
    )
    result = leave_one_dataset_out(rows)
    assert set(result.datasets) == {"d1", "d2"}


def test_none_sources_are_ignored() -> None:
    rows = [_row("healthy", None, i) for i in range(4)]
    with pytest.raises(ConfigError, match="at least two datasets"):
        leave_one_dataset_out(rows)


def test_cli_transfer_across_folders(transfer_dirs, capsys) -> None:
    rc = main(["evaluate", str(transfer_dirs[0]), str(transfer_dirs[1]), "--transfer"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "leave-one-dataset-out" in out
    assert "mean held-out accuracy" in out
    assert "datasetA" in out and "datasetB" in out


def test_cli_transfer_needs_two_folders(transfer_dirs, capsys) -> None:
    rc = main(["evaluate", str(transfer_dirs[0]), "--transfer"])
    assert rc == 2
    assert "at least two folders" in capsys.readouterr().err

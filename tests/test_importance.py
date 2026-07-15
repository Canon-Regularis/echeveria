"""Global permutation importance (Q11). Needs the ml extra."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from phytovision.analysis import AnalysisRow  # noqa: E402
from phytovision.evaluation.importance import permutation_importance  # noqa: E402
from phytovision.exceptions import ConfigError  # noqa: E402


def _row(features: dict[str, float], label: str) -> AnalysisRow:
    return AnalysisRow("x", label, None, None, 0.0, 0.0, "healthy", "m", features)


def _rows(per_class: int = 60) -> list[AnalysisRow]:
    # "signal" separates the classes; "noise" has the same distribution for both.
    rng = np.random.default_rng(0)
    rows: list[AnalysisRow] = []
    for _ in range(per_class):
        rows.append(
            _row(
                {"signal": float(rng.normal(0.40, 0.02)), "noise": float(rng.normal(0.5, 0.3))},
                "healthy",
            )
        )
        rows.append(
            _row(
                {"signal": float(rng.normal(0.30, 0.02)), "noise": float(rng.normal(0.5, 0.3))},
                "wilted",
            )
        )
    return rows


def test_importance_ranks_signal_above_noise() -> None:
    importance = dict(permutation_importance(_rows(), model="gradient-boosted"))
    assert importance["signal"] > importance["noise"]


def test_importance_needs_both_classes() -> None:
    rows = [_row({"signal": 0.4, "noise": 0.5}, "healthy") for _ in range(6)]
    with pytest.raises(ConfigError, match="both classes"):
        permutation_importance(rows)

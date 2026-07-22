"""Grouped, stratified cross-validation (F12). Requires the ``ml`` extra."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from phytovision.analysis import AnalysisRow
from phytovision.cli import main
from phytovision.evaluation.crossval import (
    _make_splits,
    grouped_stratified_cv,
)
from phytovision.exceptions import ConfigError


def _row(features: dict[str, float], label: str, source: str | None = None) -> AnalysisRow:
    return AnalysisRow(
        image_path="x",
        label=label,
        split=None,
        source=source,
        score=0.0,
        confidence=0.0,
        stress_label="healthy",
        model="m",
        features=features,
    )


def _rows(
    per_class: int = 10, seed: int = 0, sources: list[str] | None = None
) -> list[AnalysisRow]:
    """Separable two-class rows so the gradient-boosted model can actually learn."""
    rng = np.random.default_rng(seed)
    rows: list[AnalysisRow] = []
    for i in range(per_class):
        source = None if sources is None else sources[i % len(sources)]
        rows.append(
            _row(
                {"a": float(rng.normal(0.40, 0.02)), "b": float(abs(rng.normal(0.03, 0.02)))},
                "healthy",
                source,
            )
        )
        rows.append(
            _row(
                {"a": float(rng.normal(0.30, 0.02)), "b": float(abs(rng.normal(0.40, 0.05)))},
                "wilted",
                source,
            )
        )
    return rows


def test_falls_back_to_stratified_with_one_source() -> None:
    # Enough samples that each fold trains past the model's min_samples_leaf and can actually split.
    result = grouped_stratified_cv(_rows(per_class=50), n_splits=5)
    assert result.strategy == "stratified"
    assert result.n_splits == 5
    lo, hi = result.accuracy_ci95
    assert lo <= result.mean_accuracy <= hi
    assert result.mean_accuracy > 0.8  # the data is separable


def test_grouped_cv_handles_dataset_specific_features_without_leak_or_crash() -> None:
    rng = np.random.default_rng(0)
    rows: list[AnalysisRow] = []
    for _ in range(20):  # each dataset carries a feature the other lacks
        rows.append(_row({"a": float(rng.normal(0.40, 0.02)), "d1_only": 1.0}, "healthy", "d1"))
        rows.append(_row({"a": float(rng.normal(0.30, 0.02)), "d1_only": 1.0}, "wilted", "d1"))
        rows.append(_row({"a": float(rng.normal(0.41, 0.02)), "d2_only": 1.0}, "healthy", "d2"))
        rows.append(_row({"a": float(rng.normal(0.29, 0.02)), "d2_only": 1.0}, "wilted", "d2"))
    # The schema comes from the training rows per fold, so holding out a dataset whose feature is
    # absent from training no longer leaks that column into the model or crashes the fit.
    result = grouped_stratified_cv(rows, n_splits=2)
    assert result.strategy == "stratified-group"
    assert result.fold_accuracies


def test_seeded_run_is_deterministic() -> None:
    rows = _rows(per_class=50)
    first = grouped_stratified_cv(rows, n_splits=5, seed=3)
    second = grouped_stratified_cv(rows, n_splits=5, seed=3)
    assert first.fold_accuracies == second.fold_accuracies
    assert first.fold_f1s == second.fold_f1s


def test_uses_grouped_folds_with_several_sources() -> None:
    result = grouped_stratified_cv(_rows(per_class=12, sources=["d1", "d2", "d3"]), n_splits=5)
    assert result.strategy == "stratified-group"
    assert result.n_splits == 3  # capped at the number of distinct sources


def test_grouped_folds_never_share_a_source() -> None:
    labels = [0, 1] * 6
    groups = ["d1"] * 4 + ["d2"] * 4 + ["d3"] * 4
    splits, strategy = _make_splits(labels, groups, 3)
    assert strategy == "stratified-group"
    for train_idx, test_idx in splits:
        train_groups = {groups[i] for i in train_idx}
        test_groups = {groups[i] for i in test_idx}
        assert train_groups.isdisjoint(test_groups)


def test_grouped_split_tolerates_rows_with_no_source() -> None:
    # StratifiedGroupKFold sorts the group labels, which raises a TypeError on a mix of None and
    # str. Rows with no source share one synthetic group, so the split succeeds instead of crashing.
    labels = [0, 1, 0, 1, 0, 1]
    groups = ["a", "a", "b", "b", None, None]
    splits, strategy = _make_splits(labels, groups, n_splits=2)
    assert strategy == "stratified-group"
    assert len(splits) == 2


def test_needs_both_classes() -> None:
    rows = [_row({"a": 0.4, "b": 0.0}, "healthy") for _ in range(6)]
    with pytest.raises(ConfigError, match="both classes"):
        grouped_stratified_cv(rows)


def test_rejects_fewer_than_two_folds() -> None:
    with pytest.raises(ConfigError, match="at least two folds"):
        grouped_stratified_cv(_rows(), n_splits=1)


def test_rejects_too_few_samples() -> None:
    with pytest.raises(ConfigError, match="two samples"):
        grouped_stratified_cv(_rows(per_class=1), n_splits=5)


def test_cli_evaluate_cv(training_dir, capsys) -> None:
    assert main(["evaluate", str(training_dir), "--cv", "3"]) == 0
    assert "cross-validation" in capsys.readouterr().out

"""Grouped, stratified cross-validation for the water-stress model (F12).

Folds are stratified by label and, when several datasets are present, grouped by ``source`` so one
dataset never lands in both train and test. That stops the model from scoring well by memorizing
per-dataset artifacts. With only one dataset there is nothing to group on, so it falls back to plain
stratified folds. The result is reported as a mean with a confidence interval, not a single number.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from phytovision.analysis import AnalysisRow
from phytovision.evaluation._aggregate import mean_ci95
from phytovision.evaluation._common import (
    binary_labels,
    feature_keys_of,
    fit_predict_labels,
    model_factory,
)
from phytovision.evaluation.metrics import binary_metrics
from phytovision.exceptions import ConfigError

logger = logging.getLogger(__name__)

Split = tuple[Sequence[int], Sequence[int]]


@dataclass(frozen=True, slots=True)
class CrossValResult:
    """Per-fold scores plus the split strategy that produced them."""

    fold_accuracies: tuple[float, ...]
    fold_f1s: tuple[float, ...]
    strategy: str  # "stratified-group" | "stratified"

    @property
    def n_splits(self) -> int:
        return len(self.fold_accuracies)

    @property
    def mean_accuracy(self) -> float:
        return _mean(self.fold_accuracies)

    @property
    def std_accuracy(self) -> float:
        return _std(self.fold_accuracies)

    @property
    def accuracy_ci95(self) -> tuple[float, float]:
        """A 95% confidence interval for the mean accuracy (normal approximation)."""
        return mean_ci95(self.fold_accuracies)

    @property
    def mean_f1(self) -> float:
        return _mean(self.fold_f1s)


def grouped_stratified_cv(
    rows: Iterable[AnalysisRow],
    *,
    healthy_label: str = "healthy",
    n_splits: int = 5,
    model: str = "gradient-boosted",
    seed: int | None = None,
) -> CrossValResult:
    """Cross-validate a trainable model over labelled rows, grouped by ``source``.

    :param model: which trainable model to fit per fold (``gradient-boosted`` or ``ensemble``).
    :param seed: when set, shuffles the folds and seeds the model, so the run is reproducible;
        when ``None``, the folds keep their unshuffled default order.
    :raises ConfigError: if there are too few samples or classes, or the model cannot train.
    :raises ImportError: if the ``ml`` extra (scikit-learn) is not installed.
    """
    if n_splits < 2:
        raise ConfigError("cross-validation needs at least two folds")
    # resolve early so an untrainable model fails before any work
    factory = model_factory(model, seed=seed)
    rows = list(rows)
    labels = binary_labels(rows, healthy_label)
    if len(set(labels)) < 2:
        raise ConfigError("cross-validation needs both classes present in the data")

    feature_dicts = [row.features for row in rows]
    groups = [row.source for row in rows]
    splits, strategy = _make_splits(labels, groups, n_splits, seed)

    accuracies: list[float] = []
    f1s: list[float] = []
    for train_idx, test_idx in splits:
        train_labels = [labels[i] for i in train_idx]
        if len(set(train_labels)) < 2:  # a stratified fold can still be single-class on tiny data
            logger.warning("skipping a fold whose training split has a single class")
            continue
        train_dicts = [feature_dicts[i] for i in train_idx]
        # Derive the feature schema from the training rows only. Taking it over all rows leaks the
        # held-out fold's columns into the model and, when a feature is unique to that fold, breaks
        # the fit; a feature the model never trained on cannot help a held-out prediction anyway.
        keys = feature_keys_of(train_dicts)
        predictions = fit_predict_labels(
            train_dicts,
            train_labels,
            [feature_dicts[i] for i in test_idx],
            keys,
            factory,
        )
        metrics = binary_metrics([labels[i] for i in test_idx], predictions)
        accuracies.append(metrics.accuracy)
        f1s.append(metrics.f1)

    if not accuracies:
        raise ConfigError("no fold had both classes in training; cannot cross-validate")
    return CrossValResult(tuple(accuracies), tuple(f1s), strategy)


def _make_splits(
    labels: Sequence[int], groups: Sequence[str | None], n_splits: int, seed: int | None = None
) -> tuple[list[Split], str]:
    """Pick a splitter: grouped when several sources exist, otherwise plain stratified.

    A seed shuffles the folds and makes them reproducible; without one the folds keep their
    unshuffled default order, so existing fold composition does not move.
    """
    group_kfold, stratified_kfold = _splitters()
    dummy_x = np.zeros((len(labels), 1))
    distinct_groups = {group for group in groups if group is not None}
    shuffle = seed is not None

    if len(distinct_groups) >= 2:
        k = min(n_splits, len(distinct_groups))
        splitter = group_kfold(n_splits=k, shuffle=shuffle, random_state=seed)
        splits = list(splitter.split(dummy_x, labels, groups))
        return splits, "stratified-group"

    k = min(n_splits, min(labels.count(0), labels.count(1)))
    if k < 2:
        raise ConfigError("need at least two samples in the smaller class for two folds")
    splitter = stratified_kfold(n_splits=k, shuffle=shuffle, random_state=seed)
    return list(splitter.split(dummy_x, labels)), "stratified"


def _splitters() -> tuple[Any, Any]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    except ImportError as exc:  # pragma: no cover: depends on optional extra
        raise ImportError(
            "cross-validation needs the 'ml' extra: pip install -e \".[ml]\""
        ) from exc
    return StratifiedGroupKFold, StratifiedKFold


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

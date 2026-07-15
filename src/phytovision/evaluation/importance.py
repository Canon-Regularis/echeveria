"""Global permutation importance: how much a trained model relies on each feature across a dataset.

Shuffle one feature's column across samples, measure the accuracy drop, and repeat. A larger mean
drop means the model depends on that feature more. Hand-rolled to stay dependency-light and to reuse
the shared model factory, so it works for the gradient-boosted model and the ensemble.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from phytovision.analysis import AnalysisRow
from phytovision.evaluation._common import feature_keys_of, model_factory, to_plant_features
from phytovision.exceptions import ConfigError
from phytovision.models.base import StressModel


def permutation_importance(
    rows: Iterable[AnalysisRow],
    *,
    healthy_label: str = "healthy",
    model: str = "gradient-boosted",
    n_repeats: int = 5,
    seed: int = 0,
) -> list[tuple[str, float]]:
    """Rank features by mean accuracy drop when each is shuffled, most important first.

    :raises ConfigError: if both classes are not present, or the model cannot train.
    :raises ImportError: if the ``ml`` extra (scikit-learn) is not installed.
    """
    rows = list(rows)
    labels = [0 if row.label == healthy_label else 1 for row in rows]
    if len(set(labels)) < 2:
        raise ConfigError("permutation importance needs both classes present")

    feature_dicts = [row.features for row in rows]
    keys = feature_keys_of(feature_dicts)
    fitted = model_factory(model)(keys, feature_dicts, labels)
    baseline = _accuracy(fitted, feature_dicts, labels)

    rng = np.random.default_rng(seed)
    importances: dict[str, float] = {}
    for key in keys:
        drops = [
            baseline - _accuracy(fitted, _permute(feature_dicts, key, rng), labels)
            for _ in range(n_repeats)
        ]
        importances[key] = float(np.mean(drops))
    return sorted(importances.items(), key=lambda item: item[1], reverse=True)


def _accuracy(
    model: StressModel, feature_dicts: Sequence[Mapping[str, float]], labels: Sequence[int]
) -> float:
    correct = sum(
        int(model.predict(to_plant_features(row)).score >= 0.5) == label
        for row, label in zip(feature_dicts, labels, strict=True)
    )
    return correct / len(labels)


def _permute(
    feature_dicts: Sequence[Mapping[str, float]], key: str, rng: np.random.Generator
) -> list[dict[str, float]]:
    column = [row.get(key) for row in feature_dicts]
    order = rng.permutation(len(column))
    shuffled = [dict(row) for row in feature_dicts]
    for position, source in enumerate(order):
        value = column[int(source)]
        if value is None:
            shuffled[position].pop(key, None)
        else:
            shuffled[position][key] = value
    return shuffled

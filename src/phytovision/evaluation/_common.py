"""Shared building blocks for cross-validation and cross-dataset evaluation.

Both split labelled feature rows into train and test folds. Both then need the same two things: a
model trained on a fold, and its predictions turned into 0/1 labels for the metrics. Putting that
here keeps the split strategies small and testable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.types import PlantFeatures


def to_plant_features(values: Mapping[str, float]) -> PlantFeatures:
    """Wrap a flat feature dict in the ``PlantFeatures`` a model's ``predict`` expects."""
    return PlantFeatures(values=dict(values), region_count=1)


def feature_keys_of(feature_dicts: Sequence[Mapping[str, float]]) -> list[str]:
    """The sorted union of keys across rows: the schema a model trains on."""
    return sorted({key for row in feature_dicts for key in row})


def fit_predict_labels(
    train_dicts: Sequence[Mapping[str, float]],
    train_labels: Sequence[int],
    test_dicts: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
    positive_label: int = 1,
) -> list[int]:
    """Fit a model on the train fold and return 0/1 predictions for the test fold."""
    model = GradientBoostedStressModel(feature_keys, positive_label=positive_label)
    model.fit([dict(row) for row in train_dicts], list(train_labels))
    return [int(model.predict(to_plant_features(row)).score >= 0.5) for row in test_dicts]

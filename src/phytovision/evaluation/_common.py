"""Shared building blocks for cross-validation and cross-dataset evaluation.

Both split labelled feature rows into train and test folds. Both then need the same two things: a
model trained on a fold, and its predictions turned into 0/1 labels for the metrics. Putting that
here keeps the split strategies small and testable. A factory chooses the model a fold trains, so
callers can cross-validate a gradient-boosted model or an ensemble, not only the default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from phytovision.exceptions import ConfigError
from phytovision.models.base import StressModel
from phytovision.models.stress.ensemble import EnsembleStressModel
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.types import PlantFeatures

# Builds and fits a model from a fold: (feature_keys, feature_dicts, labels) -> fitted model.
ModelFactory = Callable[[Sequence[str], Sequence[Mapping[str, float]], Sequence[int]], StressModel]


def to_plant_features(values: Mapping[str, float]) -> PlantFeatures:
    """Wrap a flat feature dict in the ``PlantFeatures`` a model's ``predict`` expects."""
    return PlantFeatures(values=dict(values), region_count=1)


def feature_keys_of(feature_dicts: Sequence[Mapping[str, float]]) -> list[str]:
    """The sorted union of keys across rows: the schema a model trains on."""
    return sorted({key for row in feature_dicts for key in row})


def gradient_boosted_factory(
    feature_keys: Sequence[str],
    feature_dicts: Sequence[Mapping[str, float]],
    labels: Sequence[int],
) -> StressModel:
    model = GradientBoostedStressModel(list(feature_keys), positive_label=1)
    return model.fit([dict(row) for row in feature_dicts], list(labels))


def ensemble_factory(
    feature_keys: Sequence[str],
    feature_dicts: Sequence[Mapping[str, float]],
    labels: Sequence[int],
) -> StressModel:
    trained = gradient_boosted_factory(feature_keys, feature_dicts, labels)
    return EnsembleStressModel([HeuristicStressModel(), trained])


_FACTORIES: dict[str, ModelFactory] = {
    "gradient-boosted": gradient_boosted_factory,
    "ensemble": ensemble_factory,
}


def model_factory(name: str) -> ModelFactory:
    """Resolve a trainable model factory by name. The heuristic cannot fit, so it is not offered."""
    try:
        return _FACTORIES[name]
    except KeyError:
        raise ConfigError(
            f"{name!r} cannot be trained for evaluation; use gradient-boosted or ensemble"
        ) from None


def fit_predict_labels(
    train_dicts: Sequence[Mapping[str, float]],
    train_labels: Sequence[int],
    test_dicts: Sequence[Mapping[str, float]],
    feature_keys: Sequence[str],
    factory: ModelFactory | None = None,
) -> list[int]:
    """Fit a model on the train fold and return 0/1 predictions for the test fold."""
    build = factory or gradient_boosted_factory
    model = build(feature_keys, train_dicts, train_labels)
    return [int(model.predict(to_plant_features(row)).score >= 0.5) for row in test_dicts]

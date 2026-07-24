"""Shared building blocks for cross-validation and cross-dataset evaluation.

Both split labelled feature rows into train and test folds. Both then need the same two things: a
model trained on a fold, and its predictions turned into 0/1 labels for the metrics. Putting that
here keeps the split strategies small and testable. A factory chooses the model a fold trains, so
callers can cross-validate a gradient-boosted model or an ensemble, not only the default.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from phytovision.analysis import AnalysisRow
from phytovision.exceptions import ConfigError
from phytovision.models.base import StressModel, bucket_label
from phytovision.models.stress.ensemble import EnsembleStressModel
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.types import PlantFeatures

# Builds and fits a model from a fold: (feature_keys, feature_dicts, labels) -> fitted model.
ModelFactory = Callable[[Sequence[str], Sequence[Mapping[str, float]], Sequence[int]], StressModel]


def binary_labels(rows: Sequence[AnalysisRow], healthy_label: str) -> list[int]:
    """Map labelled rows to 0 (the healthy label) or 1 (anything else) for the metrics."""
    return [0 if row.label == healthy_label else 1 for row in rows]


def predict_label(model: StressModel, row: Mapping[str, float]) -> int:
    """A fitted model's 0/1 label for one feature row, under the package's healthy/not-healthy rule.

    The cut is the shared bucketing, not a private 0.5. The ground truth these predictions are
    scored against is "not the healthy class", and ``bucket_label`` is how every surface (the
    verdict, the API, the dashboard, and this command's own single-pass mode) decides a score is not
    healthy. A private cut measured a classifier the tool never ships, and flipped every row scoring
    in [0.33, 0.5) between ``evaluate`` and ``evaluate --cv`` on identical data.
    """
    score = model.predict(PlantFeatures.from_values(row)).score
    return int(bucket_label(score) != "healthy")


def predict_labels(model: StressModel, rows: Sequence[Mapping[str, float]]) -> list[int]:
    """A fitted model's 0/1 label for each feature row."""
    return [predict_label(model, row) for row in rows]


def feature_keys_of(feature_dicts: Sequence[Mapping[str, float]]) -> list[str]:
    """The sorted union of keys across rows: the schema a model trains on."""
    return sorted({key for row in feature_dicts for key in row})


def gradient_boosted_factory(
    feature_keys: Sequence[str],
    feature_dicts: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    *,
    seed: int | None = None,
) -> StressModel:
    model = GradientBoostedStressModel(list(feature_keys), positive_label=1, random_state=seed)
    return model.fit([dict(row) for row in feature_dicts], list(labels))


def ensemble_factory(
    feature_keys: Sequence[str],
    feature_dicts: Sequence[Mapping[str, float]],
    labels: Sequence[int],
    *,
    seed: int | None = None,
) -> StressModel:
    trained = gradient_boosted_factory(feature_keys, feature_dicts, labels, seed=seed)
    return EnsembleStressModel([HeuristicStressModel(), trained])


_FACTORIES: dict[str, Callable[..., StressModel]] = {
    "gradient-boosted": gradient_boosted_factory,
    "ensemble": ensemble_factory,
}


def trainable_model_names() -> tuple[str, ...]:
    """The model names that can be fitted, for both training and evaluation."""
    return tuple(_FACTORIES)


def model_factory(name: str, *, seed: int | None = None) -> ModelFactory:
    """Resolve a trainable model factory by name, seeded for reproducibility when ``seed`` is set.

    The heuristic cannot fit, so it is not offered.
    """
    try:
        base = _FACTORIES[name]
    except KeyError:
        raise ConfigError(
            f"{name!r} cannot be trained for evaluation; use gradient-boosted or ensemble"
        ) from None

    def build(
        feature_keys: Sequence[str],
        feature_dicts: Sequence[Mapping[str, float]],
        labels: Sequence[int],
    ) -> StressModel:
        return base(feature_keys, feature_dicts, labels, seed=seed)

    return build


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
    return predict_labels(model, test_dicts)

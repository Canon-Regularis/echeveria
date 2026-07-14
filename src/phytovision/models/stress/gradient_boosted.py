"""Gradient-boosted water-stress model (interpretable ML alternative to the heuristic).

Wraps scikit-learn's ``HistGradientBoostingClassifier`` (handles NaN features natively).
Requires the ``ml`` extra and labelled training data. It implements the ``StressModel`` contract
plus ``Trainable`` and ``ContributionModel``, so once fitted it substitutes for the heuristic. The
caller supplies ``feature_keys`` to train on — use :func:`feature_keys_from` to derive them from the
extractor stack's output so the schemas cannot drift. Contributions use a model-agnostic
baseline-substitution attribution (no SHAP required).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np

from phytovision.exceptions import ConfigError, ModelNotFittedError
from phytovision.models.base import StressModel
from phytovision.types import PlantFeatures, StressAssessment


class GradientBoostedStressModel(StressModel):
    name = "gradient-boosted-v1"

    def __init__(self, feature_keys: Sequence[str], positive_label: int = 1) -> None:
        if not feature_keys:
            raise ConfigError("feature_keys must be non-empty and fixed at construction")
        self.feature_keys = list(feature_keys)
        self.positive_label = positive_label
        self._model: object | None = None
        self._background: np.ndarray | None = None

    # --- Trainable ---
    def fit(self, feature_dicts: Sequence[dict[str, float]], labels: Sequence[int]) -> Self:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "GradientBoostedStressModel needs the 'ml' extra: pip install -e \".[ml]\""
            ) from exc

        matrix = np.array([self._vector(d) for d in feature_dicts], dtype=np.float64)
        self._background = np.nanmean(matrix, axis=0)
        model = HistGradientBoostingClassifier()
        model.fit(matrix, list(labels))
        if self.positive_label not in model.classes_:
            raise ConfigError(
                f"positive_label {self.positive_label!r} is not among the fitted classes "
                f"{list(model.classes_)}"
            )
        self._model = model
        return self

    # --- StressModel ---
    def predict(self, features: PlantFeatures) -> StressAssessment:
        x = self._vector(features.values)
        score = self._score(x)
        # Confidence: distance of the winning class probability from a coin flip.
        confidence = min(1.0, 2.0 * abs(score - 0.5))
        label = "stressed" if score >= 0.5 else "healthy"
        return StressAssessment(score, confidence, label, self.name)

    # --- ContributionModel ---
    def contributions(self, features: PlantFeatures) -> dict[str, float]:
        self._ensure_fitted()
        if self._background is None:  # set alongside _model; defensive
            raise ModelNotFittedError("model is not fitted; call fit() first")
        x = self._vector(features.values)
        base = self._score(x)
        out: dict[str, float] = {}
        for i, key in enumerate(self.feature_keys):
            perturbed = x.copy()
            perturbed[i] = self._background[i]
            out[key] = base - self._score(perturbed)  # effect of this feature vs its baseline
        return out

    def feature_label(self, key: str) -> str:
        return key

    # --- internals ---
    def _ensure_fitted(self) -> object:
        if self._model is None:
            raise ModelNotFittedError("model is not fitted; call fit() first")
        return self._model

    def _vector(self, values: Mapping[str, float | None]) -> np.ndarray:
        return np.array([_as_float(values.get(k)) for k in self.feature_keys], dtype=np.float64)

    def _score(self, x: np.ndarray) -> float:
        model = self._ensure_fitted()
        proba = model.predict_proba(x.reshape(1, -1))[0]  # type: ignore[attr-defined]
        classes = list(model.classes_)  # type: ignore[attr-defined]
        idx = classes.index(self.positive_label)  # guaranteed present: fit() checks this
        return float(proba[idx])


def _as_float(value: object) -> float:
    return float("nan") if value is None else float(value)  # type: ignore[arg-type]


def feature_keys_from(features: PlantFeatures) -> list[str]:
    """Ordered feature keys to train a model on, taken from already-extracted features.

    Bridges the extractor output schema and the model input schema: train on exactly the keys your
    extractor stack produced (``sorted(plant_features.defined())``) so the two never drift.
    """
    return sorted(features.defined())

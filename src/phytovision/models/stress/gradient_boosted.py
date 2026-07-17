"""Gradient-boosted water-stress model (interpretable ML alternative to the heuristic).

Wraps scikit-learn's ``HistGradientBoostingClassifier`` (handles NaN features natively).
Requires the ``ml`` extra and labelled training data. It implements the ``StressModel`` contract
plus ``Trainable`` and ``ContributionModel``, so once fitted it substitutes for the heuristic. The
caller supplies ``feature_keys`` to train on. Use :func:`feature_keys_from` to derive them from the
extractor stack's output so the schemas cannot drift. Contributions use a model-agnostic
baseline-substitution attribution (no SHAP required).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import numpy as np

from phytovision._num import feature_value
from phytovision.exceptions import ConfigError, ModelNotFittedError, ModelSchemaError
from phytovision.models.base import ShapResult, StressModel, bucket_label
from phytovision.types import PlantFeatures, StressAssessment

logger = logging.getLogger(__name__)


class GradientBoostedStressModel(StressModel):
    name = "gradient-boosted-v1"
    MODEL_TYPE: ClassVar[str] = "gradient-boosted"

    def __init__(
        self,
        feature_keys: Sequence[str],
        positive_label: int = 1,
        strict_schema: bool = False,
        random_state: int | None = None,
    ) -> None:
        if not feature_keys:
            raise ConfigError("feature_keys must be non-empty and fixed at construction")
        self.feature_keys = list(feature_keys)
        self.positive_label = positive_label
        # When True, predict raises on schema drift instead of vectorizing missing keys as NaN.
        self.strict_schema = strict_schema
        # Seeds the histogram learner so a fit is reproducible; None leaves sklearn's own default.
        self.random_state = random_state
        self._model: object | None = None
        self._background: np.ndarray | None = None
        self._schema_warned = False

    # --- Trainable ---
    def fit(self, feature_dicts: Sequence[dict[str, float]], labels: Sequence[int]) -> Self:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as exc:  # pragma: no cover: depends on optional extra
            raise ImportError(
                "GradientBoostedStressModel needs the 'ml' extra: pip install -e \".[ml]\""
            ) from exc

        matrix = np.array([self._vector(d) for d in feature_dicts], dtype=np.float64)
        if matrix.size == 0:
            raise ConfigError("cannot train on an empty feature matrix")
        # A feature that is missing (NaN) for every sample has nothing to bin, and the histogram
        # learner would otherwise fail with a cryptic error. Reject it with a clear message.
        all_missing = np.isnan(matrix).all(axis=0)
        if all_missing.any():
            missing = [self.feature_keys[j] for j in np.flatnonzero(all_missing)]
            raise ConfigError(f"cannot train: feature(s) {missing} are missing for every sample")
        self._background = np.nanmean(matrix, axis=0)
        model = HistGradientBoostingClassifier(random_state=self.random_state)
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
        self._check_schema(features)
        x = self._vector(features.values)
        score = self._score(x)
        # Confidence: distance of the winning class probability from a coin flip.
        confidence = min(1.0, 2.0 * abs(score - 0.5))
        # Bucket via the shared cuts so this model's verdict agrees with every other one at a score.
        label = bucket_label(score)
        return StressAssessment(score, confidence, label, self.name)

    def _check_schema(self, features: PlantFeatures) -> None:
        """Guard against extractor-stack drift: trained keys absent from the live output become NaN.

        In strict mode this raises; otherwise it warns once so the drift is visible instead of
        silently producing a confident prediction from a mostly-NaN vector.
        """
        missing = set(self.feature_keys) - set(features.defined())
        if not missing:
            return
        coverage = 1.0 - len(missing) / len(self.feature_keys)
        message = (
            f"feature schema mismatch: trained on {len(self.feature_keys)} features but "
            f"{len(missing)} are missing from the live output (coverage {coverage:.0%}); "
            f"first missing: {sorted(missing)[:5]}"
        )
        if self.strict_schema:
            raise ModelSchemaError(message)
        if not self._schema_warned:
            logger.warning("%s", message)
            self._schema_warned = True

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

    # --- ShapExplainable ---
    def shap_attribution(self, features: PlantFeatures) -> ShapResult:
        """Exact TreeSHAP attribution for one instance, in the model's margin space.

        Needs the ``ml`` extra (shap). The values, the baseline, and the model output satisfy SHAP
        completeness, so the explainer can report an additivity error close to zero.
        """
        estimator = self._ensure_fitted()
        try:
            import shap
        except ImportError as exc:  # pragma: no cover: depends on optional extra
            raise ImportError(
                "SHAP explanations need the 'ml' extra: pip install -e \".[ml]\""
            ) from exc

        x = self._vector(features.values).reshape(1, -1)
        classes = list(estimator.classes_)  # type: ignore[attr-defined]
        idx = classes.index(self.positive_label)
        explainer = shap.TreeExplainer(estimator)
        raw = explainer.shap_values(x)
        if isinstance(raw, list):  # older shap returns one array per class
            row = np.asarray(raw[idx])[0]
        else:
            arr = np.asarray(raw)
            row = arr[0, :, idx] if arr.ndim == 3 else arr[0]
        base_values = np.ravel(explainer.expected_value)
        base = float(base_values[idx] if base_values.size > 1 else base_values[0])
        output = float(np.ravel(estimator.decision_function(x))[0])  # type: ignore[attr-defined]
        return ShapResult(
            values=dict(zip(self.feature_keys, (float(v) for v in row), strict=True)),
            base_value=base,
            model_output=output,
        )

    # --- persistence ---
    def state(self) -> dict[str, object]:
        """The fitted state: schema, positive label, estimator, baseline, and drift policy."""
        self._ensure_fitted()
        return {
            "feature_keys": self.feature_keys,
            "positive_label": self.positive_label,
            "estimator": self._model,
            "background": self._background,
            "strict_schema": self.strict_schema,
            "random_state": self.random_state,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> GradientBoostedStressModel:
        model = cls(
            feature_keys=state["feature_keys"],
            positive_label=state["positive_label"],
            strict_schema=state.get("strict_schema", False),  # .get for pre-Q5 saved files
            random_state=state.get("random_state"),  # .get for files saved before seeding
        )
        model._model = state["estimator"]
        model._background = state["background"]
        return model

    def save(self, path: str | Path) -> None:
        """Persist the fitted model to ``path`` via the shared type-tagged envelope."""
        from phytovision.models.persistence import save_model

        save_model(self, path)

    @classmethod
    def load(cls, path: str | Path) -> GradientBoostedStressModel:
        """Load a model saved by :meth:`save`. The file is unpickled, so only load trusted files."""
        from phytovision.models.persistence import load_model

        model = load_model(path)
        if not isinstance(model, cls):
            raise ConfigError(f"{path} is not a {cls.__name__}")
        return model

    # --- internals ---
    def _ensure_fitted(self) -> object:
        if self._model is None:
            raise ModelNotFittedError("model is not fitted; call fit() first")
        return self._model

    def _vector(self, values: Mapping[str, float | None]) -> np.ndarray:
        nan = float("nan")
        return np.array(
            [feature_value(values, k, nan) for k in self.feature_keys], dtype=np.float64
        )

    def _score(self, x: np.ndarray) -> float:
        model = self._ensure_fitted()
        proba = model.predict_proba(x.reshape(1, -1))[0]  # type: ignore[attr-defined]
        classes = list(model.classes_)  # type: ignore[attr-defined]
        idx = classes.index(self.positive_label)  # guaranteed present: fit() checks this
        return float(proba[idx])


def feature_keys_from(features: PlantFeatures) -> list[str]:
    """Ordered feature keys to train a model on, taken from already-extracted features.

    Bridges the extractor output schema and the model input schema: train on exactly the keys your
    extractor stack produced (``sorted(plant_features.defined())``) so the two never drift.
    """
    return sorted(features.defined())

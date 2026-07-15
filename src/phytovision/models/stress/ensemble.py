"""Soft-voting ensemble stress model.

Combines several stress models by averaging their scores with optional weights. Pair the transparent
heuristic with trained models to get one score that is usually steadier than any single member.
Members that expose per-feature contributions are averaged too, so the ensemble stays explainable.
This class takes ready-built members; name-based construction lives in the registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from phytovision.exceptions import ConfigError
from phytovision.models.base import ContributionModel, StressModel, bucket_label
from phytovision.types import PlantFeatures, StressAssessment


class EnsembleStressModel(StressModel):
    name = "ensemble-v1"
    MODEL_TYPE: ClassVar[str] = "ensemble"

    def __init__(
        self, members: Sequence[StressModel], weights: Sequence[float] | None = None
    ) -> None:
        if not members:
            raise ConfigError("ensemble needs at least one member model")
        weights = [1.0] * len(members) if weights is None else list(weights)
        if len(weights) != len(members):
            raise ConfigError(
                f"weights length {len(weights)} does not match member count {len(members)}"
            )
        if any(w < 0.0 for w in weights):
            raise ConfigError("ensemble weights must be non-negative")
        total = float(sum(weights))
        if total <= 0.0:
            raise ConfigError("ensemble weights must sum to a positive value")
        self.members = list(members)
        self.weights = [w / total for w in weights]  # normalized so scores stay in [0, 1]

    def predict(self, features: PlantFeatures) -> StressAssessment:
        assessments = [m.predict(features) for m in self.members]
        pairs = list(zip(self.weights, assessments, strict=True))
        score = _clamp01(sum(w * a.score for w, a in pairs))
        confidence = _clamp01(sum(w * a.confidence for w, a in pairs))
        return StressAssessment(
            score=score,
            confidence=confidence,
            label=bucket_label(score),
            model_name=self.name,
        )

    def contributions(self, features: PlantFeatures) -> dict[str, float]:
        contributors = [
            (w, m)
            for w, m in zip(self.weights, self.members, strict=True)
            if isinstance(m, ContributionModel)
        ]
        total = float(sum(w for w, _ in contributors))
        if total <= 0.0:  # no member can attribute; nothing to explain
            return {}
        merged: dict[str, float] = {}
        for weight, model in contributors:
            share = weight / total
            for key, value in model.contributions(features).items():
                merged[key] = merged.get(key, 0.0) + share * value
        return merged

    def feature_label(self, key: str) -> str:
        for model in self.members:
            if isinstance(model, ContributionModel):
                label = model.feature_label(key)
                if label != key:
                    return label
        return key

    def state(self) -> dict[str, object]:
        """Persist each member under its own type tag, plus the normalized weights."""
        from phytovision.models.persistence import Persistable

        members: list[dict[str, object]] = []
        for model in self.members:
            if not isinstance(model, Persistable):
                raise ConfigError(f"ensemble member {type(model).__name__} cannot be saved")
            members.append({"model_type": model.MODEL_TYPE, "state": model.state()})
        return {"members": members, "weights": self.weights}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> EnsembleStressModel:
        from phytovision.models.persistence import model_from_state

        members = [model_from_state(m["model_type"], m["state"]) for m in state["members"]]
        return cls(members, weights=state["weights"])


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))

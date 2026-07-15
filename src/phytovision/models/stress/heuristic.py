"""Interpretable heuristic water-stress model, the v1 default.

A transparent, monotonic weighted model over engineered features. It needs no training, runs out of
the box, and is fully explainable (every contribution is inspectable). It is deliberately the same
interface a trained model implements, so ``GradientBoostedStressModel`` swaps in without any change
downstream. Thresholds here are documented *priors* to be replaced by a calibrated trained model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from phytovision.models.base import StressModel, bucket_label
from phytovision.types import PlantFeatures, StressAssessment


@dataclass(frozen=True)
class _Term:
    """One feature's contribution rule: normalize into [0,1] over [lo,hi], then scale by weight."""

    key: str
    lo: float
    hi: float
    weight: float  # sign = effect on stress (positive => more stress)
    label: str


# Directional priors. Greenness/turgor lower stress (negative weight); yellowing/browning,
# dullness, textural irregularity and outline concavity raise it (positive weight).
_TERMS: tuple[_Term, ...] = (
    _Term("colour.gcc_mean", 0.28, 0.42, -1.6, "green chromatic coordinate"),
    _Term("colour.exg_mean", -0.05, 0.25, -1.0, "excess green"),
    _Term("colour.greenness_ratio", 0.20, 0.95, -1.0, "fraction of green pixels"),
    _Term("colour.saturation_mean", 0.15, 0.60, -0.6, "colour saturation"),
    _Term("geometry.solidity", 0.40, 0.95, -0.8, "shape solidity (turgor)"),
    _Term("colour.yellow_fraction", 0.02, 0.50, 1.6, "yellowing"),
    _Term("colour.brown_fraction", 0.02, 0.50, 1.4, "browning / necrosis"),
    _Term("texture.entropy", 2.0, 5.0, 0.8, "surface texture entropy"),
    _Term("texture.glcm_contrast", 0.0, 5.0, 0.5, "surface contrast"),
    _Term("morphology.concavity", 0.0, 0.50, 0.6, "outline concavity (curling)"),
)


class HeuristicStressModel(StressModel):
    name = "heuristic-v1"
    MODEL_TYPE: ClassVar[str] = "heuristic"

    def __init__(
        self,
        bias: float = 0.0,
        healthy_threshold: float = 0.33,
        stressed_threshold: float = 0.66,
    ) -> None:
        self.bias = bias
        self.healthy_threshold = healthy_threshold
        self.stressed_threshold = stressed_threshold
        self._labels = {t.key: t.label for t in _TERMS}

    def state(self) -> dict[str, object]:
        """The heuristic has no fitted weights, only its bias and bucket thresholds."""
        return {
            "bias": self.bias,
            "healthy_threshold": self.healthy_threshold,
            "stressed_threshold": self.stressed_threshold,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> HeuristicStressModel:
        return cls(
            bias=state["bias"],
            healthy_threshold=state["healthy_threshold"],
            stressed_threshold=state["stressed_threshold"],
        )

    def predict(self, features: PlantFeatures) -> StressAssessment:
        contributions = self.contributions(features)
        raw = self.bias + sum(contributions.values())
        score = _sigmoid(raw)

        coverage = len(contributions) / len(_TERMS)
        # Confident when the score is decisive AND most features were available.
        confidence = min(1.0, 0.2 + 1.6 * abs(score - 0.5)) * (0.5 + 0.5 * coverage)

        return StressAssessment(
            score=score,
            confidence=confidence,
            label=bucket_label(score, self.healthy_threshold, self.stressed_threshold),
            model_name=self.name,
        )

    def contributions(self, features: PlantFeatures) -> dict[str, float]:
        defined = features.values
        out: dict[str, float] = {}
        for term in _TERMS:
            value = defined.get(term.key)
            if value is None:
                continue
            norm = _clip01((value - term.lo) / (term.hi - term.lo))
            out[term.key] = term.weight * norm
        return out

    def feature_label(self, key: str) -> str:
        return self._labels.get(key, key)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

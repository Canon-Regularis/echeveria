"""Result contract types: the stress assessment, the explanation, and the full analysis report."""

from __future__ import annotations

from dataclasses import dataclass, field

from phytovision.exceptions import ContractViolationError
from phytovision.quality import QualityAssessment
from phytovision.types.arrays import Mask
from phytovision.types.features import PlantFeatures
from phytovision.types.geometry import RegionSet


def _default_quality() -> QualityAssessment:
    """A neutral quality placeholder for a report built outside the pipeline."""
    return QualityAssessment(
        usable=True,
        flags=(),
        warnings=(),
        blur_score=0.0,
        foreground_fraction=0.0,
        luminance_std=0.0,
    )


@dataclass(frozen=True, slots=True)
class StressAssessment:
    """Output of a ``StressModel``: a bounded score plus a bounded confidence."""

    score: float  # 0 (healthy) .. 1 (severely water-stressed)
    confidence: float  # 0 .. 1
    label: str  # human-facing bucket, e.g. "healthy" / "mild" / "stressed"
    model_name: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ContractViolationError(f"stress score out of range [0,1]: {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractViolationError(f"confidence out of range [0,1]: {self.confidence}")


@dataclass(frozen=True, slots=True)
class Reason:
    """One human-readable driver of a prediction."""

    feature: str
    direction: str  # "increases" | "decreases" (effect on stress)
    contribution: float  # signed magnitude of the effect on the score
    value: float  # the feature's observed value
    description: str

    @property
    def marker(self) -> str:
        """A ``+`` when this reason raises the estimate, otherwise ``-``."""
        return "+" if self.direction == "increases" else "-"


@dataclass(frozen=True, slots=True)
class Explanation:
    """Why a model reached its verdict: the ranked reasons and the method that produced them."""

    reasons: tuple[Reason, ...]
    method: str  # e.g. "feature-contribution" | "shap"
    # How far the attribution is from completeness (sum of contributions vs the model output).
    # Set when the method has a well-defined completeness axiom (SHAP); None otherwise.
    additivity_error: float | None = None


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """The full result of running the pipeline on one image."""

    image_path: str | None
    plant_mask: Mask
    regions: RegionSet
    plant_features: PlantFeatures
    stress: StressAssessment
    explanation: Explanation
    # Outputs of optional post-model heads, keyed by head name.
    head_outputs: dict[str, object] = field(default_factory=dict)
    # Per-stage wall-clock timing in milliseconds, populated by the pipeline.
    timing_ms: dict[str, float] = field(default_factory=dict)
    # Reliability of this analysis (blur, uniformity, segmentation coverage), set by the pipeline.
    quality: QualityAssessment = field(default_factory=_default_quality)

    def summary(self) -> dict[str, object]:
        """A compact, JSON-serializable digest for CLIs / APIs."""
        digest: dict[str, object] = {
            "image_path": self.image_path,
            "region_kind": self.regions.kind,
            "region_count": len(self.regions),
            "stress": {
                "score": round(self.stress.score, 4),
                "confidence": round(self.stress.confidence, 4),
                "label": self.stress.label,
                "model": self.stress.model_name,
            },
            "explanation_method": self.explanation.method,
            "top_reasons": [
                {
                    "feature": r.feature,
                    "direction": r.direction,
                    "contribution": round(r.contribution, 4),
                    "value": round(r.value, 4),
                    "description": r.description,
                }
                for r in self.explanation.reasons[:5]
            ],
            "heads": sorted(self.head_outputs),
            "quality": self.quality.as_dict(),
        }
        if self.explanation.additivity_error is not None:
            digest["additivity_error"] = round(self.explanation.additivity_error, 5)
        if self.timing_ms:
            digest["timing_ms"] = {key: round(value, 2) for key, value in self.timing_ms.items()}
        return digest

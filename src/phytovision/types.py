"""Core, mostly-immutable data structures passed between pipeline stages.

These types are the *contract* every stage speaks. Keeping them small and validated is what lets
concrete stages be swapped freely (Liskov substitution): as long as a stage produces/consumes these
shapes correctly, the rest of the pipeline neither knows nor cares which implementation ran.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from phytovision.exceptions import ContractViolationError

# Type aliases (documentation-level; runtime is plain ``numpy.ndarray``).
Image = np.ndarray  # H x W x 3, RGB. uint8 for raw input, float32 in [0, 1] once preprocessed.
Mask = np.ndarray  # H x W, boolean. True = foreground / inside region.


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates (row/col, half-open on the max side)."""

    min_row: int
    min_col: int
    max_row: int
    max_col: int

    def __post_init__(self) -> None:
        if self.max_row <= self.min_row or self.max_col <= self.min_col:
            raise ContractViolationError(f"degenerate bbox: {self}")

    @property
    def height(self) -> int:
        return self.max_row - self.min_row

    @property
    def width(self) -> int:
        return self.max_col - self.min_col


@dataclass(frozen=True, slots=True)
class Region:
    """A single measurable region of an image (the whole plant, or later one leaf).

    Contract (upheld by every ``RegionProvider``): ``mask`` is a boolean array matching the source
    image with at least one ``True`` pixel, and ``label`` names the kind of region.
    """

    id: int
    label: str  # "plant" | "leaf" | ...
    mask: Mask
    bbox: BBox

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_:
            raise ContractViolationError(f"Region.mask must be boolean, got {self.mask.dtype}")
        if self.mask.ndim != 2:
            raise ContractViolationError(f"Region.mask must be 2-D, got shape {self.mask.shape}")
        if not self.mask.any():
            raise ContractViolationError(f"Region {self.id!r} ({self.label}) has an empty mask")

    @property
    def area_px(self) -> int:
        return int(self.mask.sum())


@dataclass(frozen=True, slots=True)
class RegionSet:
    """A non-empty, ordered collection of regions produced by a ``RegionProvider``.

    ``kind`` records what the regions represent so downstream stages behave identically whether a
    provider returned one whole-plant region or many leaf regions.
    """

    regions: tuple[Region, ...]
    kind: str  # "plant" | "leaf"
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        if not self.regions:
            raise ContractViolationError(
                "RegionSet must contain at least one region (LSP invariant)"
            )

    @property
    def is_per_leaf(self) -> bool:
        return self.kind == "leaf"

    def __iter__(self) -> Iterator[Region]:
        return iter(self.regions)

    def __len__(self) -> int:
        return len(self.regions)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Features computed for a single region. Keys are namespaced (e.g. ``geometry.area``)."""

    region_id: int
    values: dict[str, float]

    def merged_with(self, other: FeatureVector) -> FeatureVector:
        """Combine two extractors' outputs for the same region; raises on key collision."""
        if other.region_id != self.region_id:
            raise ContractViolationError(
                f"cannot merge features across regions: {self.region_id} vs {other.region_id}"
            )
        clash = self.values.keys() & other.values.keys()
        if clash:
            raise ContractViolationError(
                f"feature key collision across extractors: {sorted(clash)}"
            )
        return FeatureVector(self.region_id, {**self.values, **other.values})


@dataclass(frozen=True, slots=True)
class PlantFeatures:
    """Plant-level feature vector after aggregation.

    ``values`` may contain ``None`` for instance-only traits (e.g. ``leaf_count``) when the regions
    were not per-leaf. Those slots populate automatically once a leaf provider is used.

    Every non-null value must be finite. Extractors coerce their own output, so this invariant only
    fires on a genuine bug in a construction path that skips that coercion.
    """

    values: dict[str, float | None]
    region_count: int
    per_region: tuple[FeatureVector, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for key, value in self.values.items():
            if value is not None and not math.isfinite(value):
                raise ContractViolationError(f"feature {key!r} is not finite: {value}")

    def defined(self) -> dict[str, float]:
        """Only the non-null features, e.g. for feeding a model."""
        return {k: v for k, v in self.values.items() if v is not None}


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


@dataclass(frozen=True, slots=True)
class Explanation:
    reasons: tuple[Reason, ...]
    method: str  # e.g. "feature-contribution" | "shap"


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

    def summary(self) -> dict[str, object]:
        """A compact, JSON-serializable digest for CLIs / APIs."""
        return {
            "image_path": self.image_path,
            "region_kind": self.regions.kind,
            "region_count": len(self.regions),
            "stress": {
                "score": round(self.stress.score, 4),
                "confidence": round(self.stress.confidence, 4),
                "label": self.stress.label,
                "model": self.stress.model_name,
            },
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
        }

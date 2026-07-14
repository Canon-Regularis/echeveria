"""Model contracts.

``StressModel`` is the only thing the pipeline requires. Optional capabilities are separate
protocols, so a stage depends only on what it uses:
- ``ContributionModel`` can attribute a score to signed per-feature contributions (used by the
  explainer).
- ``Trainable`` can be fit from labelled data (used by training code, not by inference).
- ``TrainableStressModel`` is both, which is what cross-validation needs.

``Head`` is the seam for optional analyses (a future disease or temporal head) that run over
``PlantFeatures`` after the stress model, without editing the orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar, Protocol, Self, runtime_checkable

from phytovision.types import PlantFeatures, StressAssessment


class StressModel(ABC):
    name: ClassVar[str] = "stress-model"

    @abstractmethod
    def predict(self, features: PlantFeatures) -> StressAssessment: ...


def bucket_label(
    score: float, healthy_threshold: float = 0.33, stressed_threshold: float = 0.66
) -> str:
    """Map a stress score in [0, 1] to a human bucket: healthy, mild, or stressed."""
    if score < healthy_threshold:
        return "healthy"
    if score < stressed_threshold:
        return "mild"
    return "stressed"


@runtime_checkable
class ContributionModel(Protocol):
    """A model whose prediction can be split into signed per-feature contributions.

    Positive contribution pushes the score toward stressed; negative pushes toward healthy.
    """

    def contributions(self, features: PlantFeatures) -> dict[str, float]: ...

    def feature_label(self, key: str) -> str: ...


@runtime_checkable
class Trainable(Protocol):
    """A model that can be fit from labelled feature vectors. Returns self to allow chaining."""

    def fit(self, feature_dicts: Sequence[dict[str, float]], labels: Sequence[int]) -> Self: ...


@runtime_checkable
class TrainableStressModel(Protocol):
    """A model that can be both trained and used for prediction. Cross-validation needs this."""

    def fit(self, feature_dicts: Sequence[dict[str, float]], labels: Sequence[int]) -> Self: ...

    def predict(self, features: PlantFeatures) -> StressAssessment: ...


@runtime_checkable
class Head(Protocol):
    """An optional analysis that runs over ``PlantFeatures`` after the stress model.

    Attach one with ``Pipeline.add_head``. Its result appears in
    ``AnalysisReport.head_outputs[name]`` with no change to the core stages.
    """

    name: str

    def run(self, features: PlantFeatures) -> object: ...

"""Model contracts.

``StressModel`` is the only thing the pipeline requires for the primary task. Optional, *segregated*
capabilities are declared as protocols so a stage depends only on what it uses:
- ``ContributionModel`` — can attribute a score to signed per-feature contributions (used by the
  feature-contribution explainer).
- ``Trainable`` — can be fit from labelled data (used by training code, not by inference).

``Head`` is the extension seam for optional analyses (e.g. a future disease or temporal
head) that run over ``PlantFeatures`` after the stress model, without editing the orchestrator.
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


@runtime_checkable
class ContributionModel(Protocol):
    """A model whose prediction can be decomposed into signed per-feature contributions.

    Positive contribution => pushes the score toward *stressed*; negative => toward *healthy*.
    """

    def contributions(self, features: PlantFeatures) -> dict[str, float]: ...

    def feature_label(self, key: str) -> str: ...


@runtime_checkable
class Trainable(Protocol):
    """A model that can be fit from labelled feature vectors. Returns self to allow chaining."""

    def fit(self, feature_dicts: Sequence[dict[str, float]], labels: Sequence[int]) -> Self: ...


@runtime_checkable
class Head(Protocol):
    """An optional analysis that runs over ``PlantFeatures`` after the stress model.

    Heads are the Open/Closed seam for future secondary outputs (disease, temporal
    metrics): attach one with ``Pipeline.add_head`` and its result appears in
    ``AnalysisReport.head_outputs[name]`` — no change to the core stages.
    """

    name: str

    def run(self, features: PlantFeatures) -> object: ...

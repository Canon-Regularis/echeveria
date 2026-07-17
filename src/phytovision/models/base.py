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
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, runtime_checkable

from phytovision.types import PlantFeatures, StressAssessment

if TYPE_CHECKING:
    from phytovision.temporal.forecast import Forecast


class StressModel(ABC):
    name: ClassVar[str] = "stress-model"

    @abstractmethod
    def predict(self, features: PlantFeatures) -> StressAssessment: ...


# The canonical bucketing cuts on a stress score in [0, 1]: below the healthy cut is "healthy"; at
# or above the stressed cut is "stressed"; between them is "mild". Everything that references these
# cuts imports them from here, so the verdict, the forecast, and the early warning cannot drift.
HEALTHY_THRESHOLD = 0.33
STRESSED_THRESHOLD = 0.66


def bucket_label(
    score: float,
    healthy_threshold: float = HEALTHY_THRESHOLD,
    stressed_threshold: float = STRESSED_THRESHOLD,
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


@dataclass(frozen=True)
class ShapResult:
    """A SHAP attribution: per-feature values, the explainer baseline, and the model output.

    Completeness holds: ``base_value + sum(values.values())`` equals ``model_output`` up to solver
    tolerance. The values live in the model's decision (margin) space, which is fine for ranking.
    """

    values: dict[str, float]
    base_value: float
    model_output: float


@runtime_checkable
class ShapExplainable(Protocol):
    """A model that can attribute a prediction with SHAP. Needs the ``ml`` extra at call time."""

    def shap_attribution(self, features: PlantFeatures) -> ShapResult: ...


@runtime_checkable
class Trainable(Protocol):
    """A model that can be fit from labelled feature vectors. Returns self to allow chaining."""

    def fit(self, feature_dicts: Sequence[dict[str, float]], labels: Sequence[int]) -> Self: ...


@runtime_checkable
class TrainableStressModel(Protocol):
    """A model that can be both trained and used for prediction. Cross-validation needs this."""

    def fit(self, feature_dicts: Sequence[dict[str, float]], labels: Sequence[int]) -> Self: ...

    def predict(self, features: PlantFeatures) -> StressAssessment: ...


class TrajectoryForecaster(ABC):
    """A per-series stress forecaster that returns a predictive distribution, not a point.

    Where ``StressModel`` maps one image to one score, a forecaster consumes a whole stress-score
    sequence and projects it forward, reporting a mean and a prediction interval per horizon.
    Implementations live in ``models.forecasting`` and register under ``FORECASTERS``, so a caller
    selects one by name and reads every one through the same ``Forecast`` shape.
    """

    name: ClassVar[str] = "trajectory-forecaster"

    @abstractmethod
    def forecast(
        self, series: Sequence[float], horizons: Sequence[int], plant_id: str = ""
    ) -> Forecast:
        """Project ``series`` forward to each horizon as a ``Forecast`` with intervals."""


@runtime_checkable
class Head(Protocol):
    """An optional analysis that runs over ``PlantFeatures`` after the stress model.

    Attach one with ``Pipeline.add_head``. Its result appears in
    ``AnalysisReport.head_outputs[name]`` with no change to the core stages.
    """

    name: str

    def run(self, features: PlantFeatures) -> object: ...

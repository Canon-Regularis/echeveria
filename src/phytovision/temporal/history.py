"""An append-only store of per-plant observations for temporal tracking.

Each analysed image becomes an ``Observation`` keyed by ``plant_id`` and a sortable ``timestamp``.
The store groups observations by plant and returns them in time order, which is all a trend needs.
It keeps the full feature vector alongside the stress score, so any trait can be tracked over time,
not only the final verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phytovision.types import AnalysisReport


@dataclass(frozen=True, slots=True)
class Observation:
    """One plant measured at one time. ``timestamp`` is compared as text, so keep it sortable."""

    plant_id: str
    timestamp: str
    stress_score: float
    features: dict[str, float] = field(default_factory=dict)


class FeatureHistory:
    """In-memory observations grouped by plant. Insertion order is not assumed to be time order."""

    def __init__(self) -> None:
        self._by_plant: dict[str, list[Observation]] = {}

    def add(self, observation: Observation) -> None:
        self._by_plant.setdefault(observation.plant_id, []).append(observation)

    def record(self, plant_id: str, timestamp: str, report: AnalysisReport) -> Observation:
        """Build an observation from a pipeline report, store it, and return it."""
        observation = Observation(
            plant_id=plant_id,
            timestamp=timestamp,
            stress_score=report.stress.score,
            features=report.plant_features.defined(),
        )
        self.add(observation)
        return observation

    def series_for(self, plant_id: str) -> list[Observation]:
        """This plant's observations in ascending timestamp order (empty if unknown)."""
        return sorted(self._by_plant.get(plant_id, ()), key=lambda obs: obs.timestamp)

    @property
    def plant_ids(self) -> list[str]:
        return sorted(self._by_plant)

    def __len__(self) -> int:
        return sum(len(observations) for observations in self._by_plant.values())

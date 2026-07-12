"""Disease detection — reserved (future) head.

Secondary objective. A ``DiseaseModel`` consumes the SAME ``PlantFeatures`` as the stress model, so
adding disease detection is: implement this interface and register it — nothing upstream changes.
See docs/OBJECTIVES.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from phytovision.types import PlantFeatures


class DiseaseModel(ABC):
    @abstractmethod
    def predict(self, features: PlantFeatures) -> dict[str, float]:
        """Return per-disease-class probabilities."""

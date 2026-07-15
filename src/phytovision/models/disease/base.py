"""Disease detection: an optional secondary head.

A ``DiseaseModel`` consumes the same ``PlantFeatures`` as the stress model, so adding disease
detection is: implement this interface, wrap it in a ``DiseaseHead``, and register it. Nothing
upstream changes. See docs/OBJECTIVES.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from phytovision.types import PlantFeatures


class DiseaseModel(ABC):
    @abstractmethod
    def predict(self, features: PlantFeatures) -> dict[str, float]:
        """Return per-disease-class probabilities."""

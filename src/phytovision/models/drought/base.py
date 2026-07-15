"""Drought-severity staging: an optional secondary head.

A ``DroughtStageModel`` reads the same ``PlantFeatures`` as the stress model and names an ordinal
drought stage from the pattern of visible markers. It is additive: implement this interface, wrap it
in a ``DroughtStageHead``, and register it. The core healthy/mild/stressed verdict is untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from phytovision.types import PlantFeatures


class DroughtStageModel(ABC):
    @abstractmethod
    def stage(self, features: PlantFeatures) -> dict[str, object]:
        """Return the drought stage, the basis for it, and the marker scores it rests on."""

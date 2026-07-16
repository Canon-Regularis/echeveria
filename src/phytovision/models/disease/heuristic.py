"""A placeholder disease-appearance baseline. NOT a validated diagnostic.

It estimates how lesion-like the plant looks from browning and surface irregularity, purely to
demonstrate the disease-head seam end to end. It is not trained on disease data, and browning and
texture also rise under water stress, so it does not separate disease from stress. Do not use it as
a diagnosis. Replace it with a trained ``DiseaseModel`` once labelled disease data exists.
"""

from __future__ import annotations

from typing import ClassVar

from phytovision._num import clip01, feature_value
from phytovision.models.disease.base import DiseaseModel
from phytovision.types import PlantFeatures


class HeuristicDiseaseModel(DiseaseModel):
    name: ClassVar[str] = "heuristic-disease-v0"

    def predict(self, features: PlantFeatures) -> dict[str, float]:
        values = features.values
        brown = feature_value(values, "colour.brown_fraction", 0.0)
        speckle = clip01(feature_value(values, "texture.glcm_contrast", 0.0) / 5.0)
        risk = clip01(0.6 * brown + 0.4 * speckle)
        return {"healthy": 1.0 - risk, "lesion-like": risk}

"""Adapter that runs a ``DiseaseModel`` as a post-model ``Head``.

The stress pipeline attaches heads with ``Pipeline.add_head`` and stores each result in
``AnalysisReport.head_outputs[name]``. A ``DiseaseModel`` speaks ``predict`` rather than ``run``, so
this thin adapter bridges the two without the model needing to know about the head protocol.
"""

from __future__ import annotations

from phytovision.models.disease.base import DiseaseModel
from phytovision.types import PlantFeatures


class DiseaseHead:
    name = "disease"

    def __init__(self, model: DiseaseModel) -> None:
        self.model = model

    def run(self, features: PlantFeatures) -> dict[str, float]:
        return self.model.predict(features)

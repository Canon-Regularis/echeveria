"""Adapter that runs a ``DroughtStageModel`` as a post-model ``Head``.

Like the disease head, it bridges a model that speaks ``stage`` to the head protocol's ``run``, so
the pipeline can attach it with ``Pipeline.add_head`` and store the result under
``AnalysisReport.head_outputs["drought_stage"]``.
"""

from __future__ import annotations

from phytovision.models.drought.base import DroughtStageModel
from phytovision.types import PlantFeatures


class DroughtStageHead:
    name = "drought_stage"

    def __init__(self, model: DroughtStageModel) -> None:
        self.model = model

    def run(self, features: PlantFeatures) -> dict[str, object]:
        return self.model.stage(features)

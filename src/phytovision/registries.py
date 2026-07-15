"""Central component registries: the concrete wiring behind the "select by name" story.

Each built-in stage is registered here under a stable name. New implementations register themselves
the same way (``SEGMENTERS.register("my-seg")(MySegmenter)``) and become selectable via
``Pipeline.from_config`` / ``Pipeline.from_names`` and the CLI, with no edit to the orchestrator.

Registration is centralized (rather than scattered as decorators on each class) so the
stage modules stay independent of the registry, and importing any single stage does not drag in the
whole registry graph.
"""

from __future__ import annotations

from collections.abc import Sequence

from phytovision.datasets.base import DatasetLoader
from phytovision.datasets.coco import CocoDetectionLoader
from phytovision.datasets.directory import ImageDirectoryLoader
from phytovision.datasets.folder import FolderClassificationLoader
from phytovision.datasets.manifest import CsvManifestLoader
from phytovision.datasets.yolo import YoloDetectionLoader
from phytovision.explainability.base import Explainer
from phytovision.explainability.feature_reasons import FeatureContributionExplainer
from phytovision.explainability.shap_explainer import ShapExplainer
from phytovision.models.base import StressModel
from phytovision.models.disease.base import DiseaseModel
from phytovision.models.disease.heuristic import HeuristicDiseaseModel
from phytovision.models.stress.ensemble import EnsembleStressModel
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.models.stress.heuristic import HeuristicStressModel
from phytovision.phenotyping.aggregation.base import FeatureAggregator
from phytovision.phenotyping.aggregation.plant_level import PlantLevelAggregator
from phytovision.phenotyping.base import FeatureExtractor
from phytovision.phenotyping.colour import ColourFeatures
from phytovision.phenotyping.geometry import GeometryFeatures
from phytovision.phenotyping.morphology import MorphologyFeatures
from phytovision.phenotyping.texture import TextureFeatures
from phytovision.preprocessing.base import Preprocessor
from phytovision.preprocessing.basic import ResizeNormalizePreprocessor
from phytovision.regions.base import RegionProvider
from phytovision.regions.leaf_instance import LeafInstanceRegionProvider
from phytovision.regions.whole_plant import WholePlantRegionProvider
from phytovision.registry import Registry
from phytovision.segmentation.base import PlantSegmenter
from phytovision.segmentation.leaves.instance import (
    LeafInstanceSegmenter,
    NotYetTrainedLeafSegmenter,
)
from phytovision.segmentation.leaves.watershed import WatershedLeafSegmenter
from phytovision.segmentation.plant.exg_threshold import ExGThresholdSegmenter
from phytovision.segmentation.plant.lab_threshold import LabChromaSegmenter

PREPROCESSORS: Registry[Preprocessor] = Registry("preprocessor")
SEGMENTERS: Registry[PlantSegmenter] = Registry("segmenter")
LEAF_SEGMENTERS: Registry[LeafInstanceSegmenter] = Registry("leaf_segmenter")
REGION_PROVIDERS: Registry[RegionProvider] = Registry("region_provider")
FEATURE_EXTRACTORS: Registry[FeatureExtractor] = Registry("feature_extractor")
AGGREGATORS: Registry[FeatureAggregator] = Registry("aggregator")
STRESS_MODELS: Registry[StressModel] = Registry("stress_model")
EXPLAINERS: Registry[Explainer] = Registry("explainer")

PREPROCESSORS.register("resize-normalize")(ResizeNormalizePreprocessor)

SEGMENTERS.register("exg-otsu")(ExGThresholdSegmenter)
SEGMENTERS.register("lab-chroma")(LabChromaSegmenter)

LEAF_SEGMENTERS.register("watershed")(WatershedLeafSegmenter)
LEAF_SEGMENTERS.register("not-trained")(NotYetTrainedLeafSegmenter)


def _build_leaf_provider(leaf_segmenter: str = "watershed") -> LeafInstanceRegionProvider:
    """Build the leaf-instance provider with a named leaf segmenter (watershed is training-free)."""
    return LeafInstanceRegionProvider(LEAF_SEGMENTERS.create(leaf_segmenter))


REGION_PROVIDERS.register("whole-plant")(WholePlantRegionProvider)
REGION_PROVIDERS.register("leaf-instance")(_build_leaf_provider)

FEATURE_EXTRACTORS.register("geometry")(GeometryFeatures)
FEATURE_EXTRACTORS.register("colour")(ColourFeatures)
FEATURE_EXTRACTORS.register("texture")(TextureFeatures)
FEATURE_EXTRACTORS.register("morphology")(MorphologyFeatures)

AGGREGATORS.register("plant-level")(PlantLevelAggregator)

STRESS_MODELS.register("heuristic")(HeuristicStressModel)
STRESS_MODELS.register("gradient-boosted")(GradientBoostedStressModel)


def _build_ensemble(
    members: Sequence[str] = ("heuristic",), weights: Sequence[float] | None = None
) -> EnsembleStressModel:
    """Build an ensemble from member model names. Name resolution stays in the composition root."""
    built = [STRESS_MODELS.create(name) for name in members]
    return EnsembleStressModel(built, weights=weights)


STRESS_MODELS.register("ensemble")(_build_ensemble)

EXPLAINERS.register("feature-contribution")(FeatureContributionExplainer)
EXPLAINERS.register("shap")(ShapExplainer)

# Optional secondary heads. The shipped disease model is an unvalidated placeholder (see its docs).
DISEASE_MODELS: Registry[DiseaseModel] = Registry("disease_model")
DISEASE_MODELS.register("heuristic")(HeuristicDiseaseModel)

# Dataset loaders are selectable by name too. Their first constructor argument is the path (dataset
# root, annotations file, or manifest), so callers pass it as the loader's own keyword.
DATASET_LOADERS: Registry[DatasetLoader] = Registry("dataset_loader")
DATASET_LOADERS.register("folder")(FolderClassificationLoader)
DATASET_LOADERS.register("directory")(ImageDirectoryLoader)
DATASET_LOADERS.register("coco")(CocoDetectionLoader)
DATASET_LOADERS.register("csv")(CsvManifestLoader)
DATASET_LOADERS.register("yolo")(YoloDetectionLoader)

# Default component names used by Pipeline.default() / from_config() when a slot is unspecified.
DEFAULTS = {
    "preprocessor": "resize-normalize",
    "segmenter": "exg-otsu",
    "region_provider": "whole-plant",
    "feature_extractors": ("geometry", "colour", "texture", "morphology"),
    "aggregator": "plant-level",
    "model": "heuristic",
    "explainer": "feature-contribution",
}

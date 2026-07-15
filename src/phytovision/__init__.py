"""PhytoVision, an explainable computer-vision framework for plant phenotyping and water-stress
detection. See docs/ARCHITECTURE.md for the design.

The stable public surface is re-exported here: the ``Pipeline`` orchestrator, the core data types
every stage speaks, the stage interfaces you implement, and the exception hierarchy.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from phytovision.exceptions import (
    ConfigError,
    ContractViolationError,
    InvalidImageError,
    ModelNotFittedError,
    ModelSchemaError,
    PhytoVisionError,
    SegmentationError,
)
from phytovision.explainability.base import Explainer
from phytovision.models.base import ContributionModel, Head, StressModel, Trainable
from phytovision.phenotyping.aggregation.base import FeatureAggregator
from phytovision.phenotyping.base import FeatureExtraction, FeatureExtractor
from phytovision.pipeline import Pipeline
from phytovision.preprocessing.base import Preprocessor
from phytovision.regions.base import RegionProvider
from phytovision.segmentation.base import PlantSegmenter
from phytovision.types import (
    AnalysisReport,
    BBox,
    Explanation,
    FeatureVector,
    Image,
    Mask,
    PlantFeatures,
    Reason,
    Region,
    RegionSet,
    StressAssessment,
)

try:
    __version__ = version("phytovision")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # orchestration
    "Pipeline",
    # core data types
    "AnalysisReport",
    "BBox",
    "Explanation",
    "FeatureVector",
    "Image",
    "Mask",
    "PlantFeatures",
    "Reason",
    "Region",
    "RegionSet",
    "StressAssessment",
    # stage interfaces (implement one to add a component)
    "Preprocessor",
    "PlantSegmenter",
    "RegionProvider",
    "FeatureExtraction",
    "FeatureExtractor",
    "FeatureAggregator",
    "StressModel",
    "ContributionModel",
    "Trainable",
    "Head",
    "Explainer",
    # exceptions
    "PhytoVisionError",
    "InvalidImageError",
    "ContractViolationError",
    "SegmentationError",
    "ModelNotFittedError",
    "ModelSchemaError",
    "ConfigError",
]

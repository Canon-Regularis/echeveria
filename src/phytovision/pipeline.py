"""The pipeline orchestrator.

``Pipeline`` depends only on the abstract stage interfaces (dependency inversion); concrete
implementations are injected. Build one three ways:

- ``Pipeline.default()``: the v1 stack.
- ``Pipeline.from_names(model=..., ...)``: swap stages by registered name.
- ``Pipeline.from_config({...})``: full config with names and params, loaded from TOML/JSON.

The ``with_*`` builders return a new pipeline with one stage swapped; ``add_head`` adds an optional
post-model head. Nothing here is edited to add a new implementation: it registers in
:mod:`phytovision.registries` and becomes selectable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

import numpy as np

from phytovision.config_schema import ComponentSpec, PipelineConfig
from phytovision.exceptions import ConfigError, InvalidImageError
from phytovision.explainability.base import Explainer
from phytovision.io import load_image
from phytovision.models.base import Head, StressModel
from phytovision.phenotyping.aggregation.base import FeatureAggregator
from phytovision.phenotyping.base import CompositeFeatureExtractor, FeatureExtraction
from phytovision.preprocessing.base import Preprocessor
from phytovision.quality import assess_quality
from phytovision.regions.base import RegionProvider
from phytovision.registries import (
    AGGREGATORS,
    EXPLAINERS,
    FEATURE_EXTRACTORS,
    PREPROCESSORS,
    REGION_PROVIDERS,
    SEGMENTERS,
    STRESS_MODELS,
)
from phytovision.registry import Registry
from phytovision.segmentation.base import PlantSegmenter
from phytovision.types import AnalysisReport, Image
from phytovision.validation import validate_rgb_image

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Pipeline:
    preprocessor: Preprocessor
    segmenter: PlantSegmenter
    region_provider: RegionProvider
    feature_extractor: FeatureExtraction
    aggregator: FeatureAggregator
    model: StressModel
    explainer: Explainer
    heads: tuple[Head, ...] = ()

    @classmethod
    def default(cls) -> Pipeline:
        """The v1 wiring: classical segmentation, whole-plant regions, heuristic stress model."""
        return cls.from_config({})

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None = None) -> Pipeline:
        """Build a pipeline from a config mapping; unspecified slots use the registered defaults.

        :param config: keys ``preprocessor``, ``segmenter``, ``region_provider``,
            ``feature_extractors``, ``aggregator``, ``model``, ``explainer``. Each value is a
            registered name, or ``{"name": ..., "params": {...}}``. A spec with only ``params``
            keeps the slot's default component and just overrides its parameters. An unknown
            top-level key is rejected, so a mistyped slot name fails loudly instead of silently.
        :raises ConfigError: if a top-level key is unknown, a component name is unknown, or a spec
            is malformed.
        """
        resolved = PipelineConfig.from_mapping(config)
        extractors = [_create(FEATURE_EXTRACTORS, spec) for spec in resolved.feature_extractors]
        return cls(
            preprocessor=_create(PREPROCESSORS, resolved.preprocessor),
            segmenter=_create(SEGMENTERS, resolved.segmenter),
            region_provider=_create(REGION_PROVIDERS, resolved.region_provider),
            feature_extractor=CompositeFeatureExtractor(extractors),
            aggregator=_create(AGGREGATORS, resolved.aggregator),
            model=_create(STRESS_MODELS, resolved.model),
            explainer=_create(EXPLAINERS, resolved.explainer),
        )

    @classmethod
    def from_names(cls, **names: object) -> Pipeline:
        """Convenience wrapper over :meth:`from_config` for parameter-less name swaps."""
        return cls.from_config(names)

    def analyze(self, image: str | Path | Image) -> AnalysisReport:
        """Run the full pipeline on one image and return an :class:`AnalysisReport`.

        :param image: a path to an image file, or an ``H x W x 3`` uint8/float RGB ndarray.
        :returns: the segmentation mask, regions, plant features, stress, explanation, and
            any head outputs.
        :raises InvalidImageError: if ``image`` is not a valid RGB image.
        :raises FileNotFoundError: if a path is given that does not exist.
        """
        started = time.perf_counter()
        image_path: str | None = None
        if isinstance(image, (str, Path)):
            image_path = str(image)
            raw = load_image(image)
        elif isinstance(image, np.ndarray):
            raw = image
        else:
            raise InvalidImageError(
                f"analyze() expects a path or ndarray, got {type(image).__name__}"
            )
        validate_rgb_image(raw)
        logger.debug("analyze: input=%s shape=%s", image_path or "<ndarray>", raw.shape)

        timing_ms: dict[str, float] = {}
        mark = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal mark
            now = time.perf_counter()
            timing_ms[stage] = (now - mark) * 1000.0
            mark = now

        prepared = self.preprocessor.process(raw)
        lap("preprocess")
        plant_mask = self.segmenter.segment(prepared)
        lap("segment")
        regions = self.region_provider.regions(prepared, plant_mask)
        lap("regions")
        logger.debug(
            "analyze: %d region(s) kind=%s coverage=%.3f",
            len(regions),
            regions.kind,
            float(plant_mask.mean()),
        )

        features = [self.feature_extractor.extract(prepared, region) for region in regions]
        plant_features = self.aggregator.aggregate(
            regions, features, reduction_policy=self.feature_extractor.reduction_policy()
        )
        lap("extract")
        assessment = self.model.predict(plant_features)
        lap("model")
        explanation = self.explainer.explain(self.model, plant_features, assessment)
        lap("explain")

        head_outputs: dict[str, object] = {}
        for head in self.heads:
            head_outputs[head.name] = head.run(plant_features)
        if self.heads:
            lap("heads")

        quality = assess_quality(prepared, float(plant_mask.mean()))
        lap("quality")

        timing_ms["total"] = (time.perf_counter() - started) * 1000.0
        logger.debug(
            "analyze: stress=%.3f (%s) in %.1f ms",
            assessment.score,
            assessment.label,
            timing_ms["total"],
        )
        return AnalysisReport(
            image_path=image_path,
            plant_mask=plant_mask,
            regions=regions,
            plant_features=plant_features,
            stress=assessment,
            explanation=explanation,
            head_outputs=head_outputs,
            timing_ms=timing_ms,
            quality=quality,
        )

    # --- builders (return a new Pipeline with one stage swapped / a head added) ---
    def with_preprocessor(self, preprocessor: Preprocessor) -> Pipeline:
        """Return a copy using ``preprocessor``."""
        return replace(self, preprocessor=preprocessor)

    def with_segmenter(self, segmenter: PlantSegmenter) -> Pipeline:
        """Return a copy using ``segmenter``."""
        return replace(self, segmenter=segmenter)

    def with_region_provider(self, region_provider: RegionProvider) -> Pipeline:
        """Return a copy using ``region_provider`` (e.g. swap whole-plant for leaf-instance)."""
        return replace(self, region_provider=region_provider)

    def with_feature_extractor(self, feature_extractor: FeatureExtraction) -> Pipeline:
        """Return a copy using ``feature_extractor``."""
        return replace(self, feature_extractor=feature_extractor)

    def with_aggregator(self, aggregator: FeatureAggregator) -> Pipeline:
        """Return a copy using ``aggregator``."""
        return replace(self, aggregator=aggregator)

    def with_model(self, model: StressModel) -> Pipeline:
        """Return a copy using ``model``."""
        return replace(self, model=model)

    def with_explainer(self, explainer: Explainer) -> Pipeline:
        """Return a copy using ``explainer``."""
        return replace(self, explainer=explainer)

    def add_head(self, head: Head) -> Pipeline:
        """Return a copy with an additional post-model head attached."""
        return replace(self, heads=(*self.heads, head))


def _create(registry: Registry[_T], spec: ComponentSpec) -> _T:
    """Instantiate a component from a resolved ``ComponentSpec`` via ``registry``.

    The spec is already validated and normalized by ``PipelineConfig``; this only resolves the name
    against the registry and surfaces a clean ``ConfigError`` for an unknown name or bad parameters.
    """
    try:
        return registry.create(spec.name, **spec.params)
    except KeyError as exc:
        # KeyError.__str__ repr-quotes its message, so read the raw text to keep the error clean.
        message = str(exc.args[0]) if exc.args else str(exc)
        raise ConfigError(message) from exc
    except TypeError as exc:
        raise ConfigError(f"could not construct {spec.name!r}: {exc}") from exc

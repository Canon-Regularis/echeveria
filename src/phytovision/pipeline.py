"""The pipeline orchestrator.

``Pipeline`` depends only on the abstract stage interfaces (dependency inversion); concrete
implementations are injected. Build one three ways:

- ``Pipeline.default()`` — the v1 stack.
- ``Pipeline.from_names(model=..., ...)`` — swap stages by registered name.
- ``Pipeline.from_config({...})`` — full config (names + params), e.g. loaded from TOML/JSON.

The ``with_*`` builders return a new pipeline with one stage swapped; ``add_head`` adds an optional
post-model head. Nothing here is edited to add a new implementation — it registers in
:mod:`phytovision.registries` and becomes selectable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

import numpy as np

from phytovision.exceptions import ConfigError, InvalidImageError
from phytovision.explainability.base import Explainer
from phytovision.io import load_image
from phytovision.models.base import Head, StressModel
from phytovision.phenotyping.aggregation.base import FeatureAggregator
from phytovision.phenotyping.base import CompositeFeatureExtractor, FeatureExtraction
from phytovision.preprocessing.base import Preprocessor
from phytovision.regions.base import RegionProvider
from phytovision.registries import (
    AGGREGATORS,
    DEFAULTS,
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

# A component spec is either a registered name, or {"name": ..., "params": {...}}.
ComponentSpec = str | Mapping[str, object]


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
            keeps the slot's default component and just overrides its parameters.
        :raises ConfigError: if a component name is unknown or a spec is malformed.
        """
        cfg = dict(config or {})
        extractor_specs = cfg.get("feature_extractors", DEFAULTS["feature_extractors"])
        if not isinstance(extractor_specs, Sequence) or isinstance(extractor_specs, str):
            raise ConfigError("`feature_extractors` must be a list of component specs")
        extractors = [_build(FEATURE_EXTRACTORS, s) for s in extractor_specs]

        def slot(name: str, registry: Registry[_T]) -> _T:
            default = DEFAULTS[name]
            return _build(registry, cfg.get(name, default), default_name=default)

        return cls(
            preprocessor=slot("preprocessor", PREPROCESSORS),
            segmenter=slot("segmenter", SEGMENTERS),
            region_provider=slot("region_provider", REGION_PROVIDERS),
            feature_extractor=CompositeFeatureExtractor(extractors),
            aggregator=slot("aggregator", AGGREGATORS),
            model=slot("model", STRESS_MODELS),
            explainer=slot("explainer", EXPLAINERS),
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

        prepared = self.preprocessor.process(raw)
        plant_mask = self.segmenter.segment(prepared)
        regions = self.region_provider.regions(prepared, plant_mask)
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
        assessment = self.model.predict(plant_features)
        explanation = self.explainer.explain(self.model, plant_features, assessment)

        head_outputs: dict[str, object] = {}
        for head in self.heads:
            head_outputs[head.name] = head.run(plant_features)

        logger.debug(
            "analyze: stress=%.3f (%s) in %.1f ms",
            assessment.score,
            assessment.label,
            (time.perf_counter() - started) * 1000.0,
        )
        return AnalysisReport(
            image_path=image_path,
            plant_mask=plant_mask,
            regions=regions,
            plant_features=plant_features,
            stress=assessment,
            explanation=explanation,
            head_outputs=head_outputs,
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


def _build(registry: Registry[_T], spec: object, default_name: object = None) -> _T:
    """Instantiate a component from a name or ``{"name", "params"}`` spec via ``registry``.

    If a mapping spec omits ``name``, ``default_name`` is used, so a config can override just the
    parameters of a slot's default component.
    """
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, Mapping):
        raw_name = spec.get("name", default_name)
        if not isinstance(raw_name, str):
            raise ConfigError(f"component spec needs a string 'name': {spec!r}")
        name = raw_name
        raw_params = spec.get("params", {})
        params = dict(raw_params) if isinstance(raw_params, Mapping) else {}
    else:
        raise ConfigError(f"invalid component spec: {spec!r}")
    try:
        return registry.create(name, **params)
    except KeyError as exc:
        raise ConfigError(str(exc)) from exc
    except TypeError as exc:
        raise ConfigError(f"could not construct {name!r}: {exc}") from exc

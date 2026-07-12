"""Each extractor honours the shared contract: namespaced keys, finite floats, no side effects."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phytovision.phenotyping.base import CompositeFeatureExtractor, FeatureExtractor
from phytovision.phenotyping.colour import ColourFeatures
from phytovision.phenotyping.geometry import GeometryFeatures
from phytovision.phenotyping.morphology import MorphologyFeatures
from phytovision.phenotyping.texture import TextureFeatures

_EXTRACTORS = [GeometryFeatures, ColourFeatures, TextureFeatures, MorphologyFeatures]


@pytest.mark.parametrize("extractor_cls", _EXTRACTORS)
def test_extractor_output_is_namespaced_and_finite(
    extractor_cls, healthy_image, plant_region
) -> None:
    extractor: FeatureExtractor = extractor_cls()
    fv = extractor.extract(healthy_image, plant_region)

    assert fv.region_id == plant_region.id
    assert fv.values, "extractor produced no features"
    for key, value in fv.values.items():
        assert key.startswith(f"{extractor.namespace}."), key
        assert isinstance(value, float) and math.isfinite(value), (key, value)


def test_extractors_do_not_mutate_input(healthy_image, plant_region) -> None:
    before = healthy_image.copy()
    for cls in _EXTRACTORS:
        cls().extract(healthy_image, plant_region)
    assert np.array_equal(healthy_image, before)


def test_composite_merges_without_collision(healthy_image, plant_region) -> None:
    composite = CompositeFeatureExtractor([cls() for cls in _EXTRACTORS])
    fv = composite.extract(healthy_image, plant_region)

    namespaces = {key.split(".", 1)[0] for key in fv.values}
    assert namespaces == {"geometry", "colour", "texture", "morphology"}


def test_greenness_features_separate_healthy_from_stressed(
    healthy_image, stressed_image, plant_region
) -> None:
    colour = ColourFeatures()
    healthy = colour.extract(healthy_image, plant_region).values
    stressed = colour.extract(stressed_image, plant_region).values

    assert healthy["colour.gcc_mean"] > stressed["colour.gcc_mean"]
    assert healthy["colour.yellow_fraction"] < stressed["colour.yellow_fraction"]

"""Extractor-specific behaviour beyond the shared contract (which lives in tests/contracts/)."""

from __future__ import annotations

from phytovision.phenotyping.base import CompositeFeatureExtractor
from phytovision.phenotyping.colour import ColourFeatures
from phytovision.phenotyping.geometry import GeometryFeatures
from phytovision.phenotyping.morphology import MorphologyFeatures
from phytovision.phenotyping.texture import TextureFeatures

_EXTRACTORS = [GeometryFeatures, ColourFeatures, TextureFeatures, MorphologyFeatures]


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

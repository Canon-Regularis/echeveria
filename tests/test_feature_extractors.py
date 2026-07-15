"""Extractor-specific behaviour beyond the shared contract (which lives in tests/contracts/)."""

from __future__ import annotations

import numpy as np

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


def test_red_fraction_flags_anthocyanin_reddening(plant_region) -> None:
    colour = ColourFeatures()
    red = np.zeros((128, 128, 3), np.float32)
    red[..., 0] = 0.7  # strongly red foreground (hue ~0) -> anthocyanin band
    magenta = np.zeros((128, 128, 3), np.float32)
    magenta[..., 0] = 0.7  # R
    magenta[..., 2] = 0.7  # B -> magenta (hue ~0.83), the wraparound anthocyanin band
    green = np.zeros((128, 128, 3), np.float32)
    green[..., 1] = 0.6  # green foreground

    red_frac = colour.extract(red, plant_region).values["colour.red_fraction"]
    magenta_frac = colour.extract(magenta, plant_region).values["colour.red_fraction"]
    green_frac = colour.extract(green, plant_region).values["colour.red_fraction"]
    assert 0.0 <= green_frac <= red_frac <= 1.0
    assert red_frac > 0.5  # most of a red plant reads as reddened
    assert magenta_frac > 0.5  # the hue>=0.80 purple/magenta band is covered too

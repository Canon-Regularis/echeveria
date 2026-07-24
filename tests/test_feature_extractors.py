"""Extractor-specific behaviour beyond the shared contract (which lives in tests/contracts/)."""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.phenotyping.base import CompositeFeatureExtractor
from phytovision.phenotyping.colour import ColourFeatures, circular_hue_mean
from phytovision.phenotyping.geometry import GeometryFeatures
from phytovision.phenotyping.morphology import MorphologyFeatures
from phytovision.phenotyping.texture import TextureFeatures
from phytovision.regions.base import region_from_mask

_EXTRACTORS = [GeometryFeatures, ColourFeatures, TextureFeatures, MorphologyFeatures]


def test_composite_merges_without_collision(healthy_image, plant_region) -> None:
    composite = CompositeFeatureExtractor([cls() for cls in _EXTRACTORS])
    fv = composite.extract(healthy_image, plant_region)

    namespaces = {key.split(".", 1)[0] for key in fv.values}
    assert namespaces == {"geometry", "colour", "texture", "morphology"}


def test_circular_hue_mean_stays_below_one_at_the_seam() -> None:
    # Two reds straddling the wraparound average to a mean vector a hair below the seam, and that
    # negative angle's modulo rounds up to exactly 1.0 in float: outside the documented [0, 1) and
    # the far end of the linear feature range from the ~0.0 it should read as.
    mean = circular_hue_mean(np.array([0.02, 0.98]))
    assert 0.0 <= mean < 1.0
    assert min(mean, 1.0 - mean) < 0.01  # sits on the red seam, not the opposite hue


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


def test_excess_green_is_exposure_invariant() -> None:
    # ExG is defined on chromatic coordinates (the segmenter says so and computes it that way, and
    # the heuristic's term range assumes that scale). Computed on raw intensities it scaled with
    # brightness, so the same pigment photographed brighter read greener and moved the stress score.
    mask = np.ones((16, 16), dtype=bool)
    region = region_from_mask(0, "plant", mask)
    values = []
    for exposure in (0.6, 1.2, 2.0):
        image = np.zeros((16, 16, 3), np.float32)
        image[..., 0] = 0.33 * exposure
        image[..., 1] = 0.38 * exposure
        image[..., 2] = 0.29 * exposure
        values.append(ColourFeatures().extract(image, region).values["colour.exg_mean"])
    assert max(values) - min(values) < 1e-6  # identical pigment, identical index
    assert values[0] == pytest.approx(2 * 0.38 - 0.33 - 0.29, abs=1e-6)

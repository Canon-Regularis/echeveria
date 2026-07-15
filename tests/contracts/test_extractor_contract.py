"""Every registered feature extractor upholds the FeatureExtractor contract (registry-driven)."""

from __future__ import annotations

import numpy as np
import pytest
from _invariants import assert_valid_feature_vector
from _strategies import images_with_region
from hypothesis import given, settings

from phytovision.registries import FEATURE_EXTRACTORS

_NAMES = FEATURE_EXTRACTORS.names()


def test_expected_extractors_are_registered() -> None:
    # Registry-driven parametrize auto-enrolls new ones; this guards against losing a built-in.
    assert {"geometry", "colour", "texture", "morphology"} <= set(_NAMES)


@pytest.mark.parametrize("name", _NAMES)
def test_extractor_output_is_namespaced_and_finite(name, healthy_image, plant_region) -> None:
    extractor = FEATURE_EXTRACTORS.create(name)
    vector = extractor.extract(healthy_image, plant_region)
    assert vector.region_id == plant_region.id
    assert_valid_feature_vector(vector, extractor.namespace)


@pytest.mark.parametrize("name", _NAMES)
def test_extractor_does_not_mutate_its_input(name, healthy_image, plant_region) -> None:
    image_before = healthy_image.copy()
    mask_before = plant_region.mask.copy()
    FEATURE_EXTRACTORS.create(name).extract(healthy_image, plant_region)
    assert np.array_equal(healthy_image, image_before)
    assert np.array_equal(plant_region.mask, mask_before)


@pytest.mark.parametrize("name", _NAMES)
@given(sample=images_with_region())
@settings(max_examples=20, deadline=None)
def test_extractor_stays_finite_on_random_regions(name, sample) -> None:
    # The finiteness contract must hold on adversarial regions (uniform patches, single pixels). It
    # guards the coercion in FeatureExtractor.extract: without it a _compute returning NaN or inf
    # (say a GLCM correlation on a uniform patch) would leak through and fail this test.
    image, region = sample
    extractor = FEATURE_EXTRACTORS.create(name)
    assert_valid_feature_vector(extractor.extract(image, region), extractor.namespace)

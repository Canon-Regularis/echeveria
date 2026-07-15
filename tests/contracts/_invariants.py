"""Shared invariants every registered component must satisfy.

These are the substitutability (Liskov) checks: any implementation of a stage contract must uphold
them, so the contract tests parametrize them off the registries. Registering a component enrolls it
in these checks automatically.
"""

from __future__ import annotations

import math

import numpy as np

from phytovision.types import FeatureVector, Image, Mask, RegionSet, StressAssessment

_BUCKETS = {"healthy", "mild", "stressed"}


def assert_valid_mask(mask: Mask, image: Image) -> None:
    """A segmenter must return an image-sized, non-empty boolean mask."""
    assert mask.dtype == np.bool_, mask.dtype
    assert mask.shape == image.shape[:2], (mask.shape, image.shape)
    assert mask.any(), "segmenter returned an empty mask (fallbacks should prevent this)"


def assert_valid_assessment(assessment: StressAssessment) -> None:
    """A stress model must return bounded scores and a known bucket label."""
    assert isinstance(assessment, StressAssessment)
    assert 0.0 <= assessment.score <= 1.0, assessment.score
    assert 0.0 <= assessment.confidence <= 1.0, assessment.confidence
    assert assessment.label in _BUCKETS, assessment.label


def assert_valid_feature_vector(vector: FeatureVector, namespace: str | None = None) -> None:
    """An extractor must return namespaced, finite float values."""
    assert vector.values, "extractor produced no features"
    for key, value in vector.values.items():
        if namespace is not None:
            assert key.startswith(f"{namespace}."), key
        assert isinstance(value, float) and math.isfinite(value), (key, value)


def assert_valid_region_set(region_set: RegionSet, image_shape: tuple[int, int]) -> None:
    """A region provider must return a non-empty set of image-sized boolean masks."""
    assert isinstance(region_set, RegionSet)
    assert len(region_set) >= 1
    assert region_set.image_shape == image_shape, (region_set.image_shape, image_shape)
    for region in region_set:
        assert region.mask.dtype == np.bool_
        assert region.mask.shape == image_shape
        assert region.mask.any()

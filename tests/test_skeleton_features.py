"""The skeleton silhouette-morphology feature extractor."""

from __future__ import annotations

import math

import numpy as np

from phytovision.phenotyping.base import CompositeFeatureExtractor
from phytovision.phenotyping.geometry import GeometryFeatures
from phytovision.phenotyping.skeleton import SkeletonFeatures
from phytovision.regions.base import region_from_mask
from phytovision.registries import DEFAULTS, FEATURE_EXTRACTORS
from phytovision.types import Region

_KEYS = {
    "skeleton.length",
    "skeleton.length_to_area",
    "skeleton.branch_points",
    "skeleton.endpoints",
    "skeleton.mean_thickness",
    "skeleton.tortuosity",
}


def _blob(size: int = 64, ry: float = 25.0, rx: float = 150.0) -> Region:
    yy, xx = np.mgrid[0:size, 0:size]
    mask = (yy - size / 2) ** 2 / ry + (xx - size / 2) ** 2 / rx <= 1.0
    return region_from_mask(0, "leaf", mask)


def _image(size: int = 64) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.float32)


def test_skeleton_descriptors_are_finite_and_namespaced() -> None:
    values = SkeletonFeatures().extract(_image(), _blob()).values
    assert set(values) == _KEYS
    assert all(math.isfinite(v) for v in values.values())
    assert values["skeleton.length"] > 0.0
    assert values["skeleton.mean_thickness"] > 0.0


def test_a_tiny_mask_returns_zeros_without_crashing() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = mask[2, 3] = True
    values = SkeletonFeatures().extract(_image(5), region_from_mask(0, "leaf", mask)).values
    assert set(values) == _KEYS
    assert all(v == 0.0 for v in values.values())


def test_skeleton_is_registered_but_not_a_default() -> None:
    assert "skeleton" in FEATURE_EXTRACTORS.names()
    assert isinstance(FEATURE_EXTRACTORS.create("skeleton"), SkeletonFeatures)
    # It must not be in the default stack, so the shipped feature schema does not drift.
    assert "skeleton" not in DEFAULTS["feature_extractors"]


def test_skeleton_does_not_change_other_namespaces() -> None:
    # Adding the skeleton extractor to a stack leaves the other extractors' values byte for byte.
    region = _blob()
    image = _image()
    geometry_alone = GeometryFeatures().extract(image, region).values
    combined = (
        CompositeFeatureExtractor([GeometryFeatures(), SkeletonFeatures()])
        .extract(image, region)
        .values
    )
    geometry_from_combined = {k: v for k, v in combined.items() if k.startswith("geometry.")}
    assert geometry_from_combined == geometry_alone
    assert set(combined) >= _KEYS  # and the skeleton keys were added


def test_extensive_traits_are_summed_by_policy() -> None:
    policy = SkeletonFeatures().reduction_policy()
    assert policy["skeleton.length"] == "sum"
    assert policy["skeleton.branch_points"] == "sum"
    assert policy["skeleton.endpoints"] == "sum"
    assert "skeleton.mean_thickness" not in policy  # intensive traits are area-weighted-averaged

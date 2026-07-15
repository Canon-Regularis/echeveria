"""Runtime feature contract: PlantFeatures enforces finiteness; ranges are declared and checked."""

from __future__ import annotations

import pytest

from phytovision.exceptions import ContractViolationError
from phytovision.feature_contract import range_violations
from phytovision.pipeline import Pipeline
from phytovision.regions.leaf_instance import LeafInstanceRegionProvider
from phytovision.types import PlantFeatures


def test_plant_features_rejects_non_finite() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ContractViolationError, match="not finite"):
            PlantFeatures(values={"x": bad}, region_count=1)


def test_plant_features_allows_none_and_finite() -> None:
    features = PlantFeatures(values={"a": 0.5, "b": None}, region_count=1)
    assert features.defined() == {"a": 0.5}


def test_range_violations_detects_out_of_range() -> None:
    assert range_violations({"colour.gcc_mean": 1.5}) == {"colour.gcc_mean": 1.5}
    assert range_violations({"colour.gcc_mean": 0.4}) == {}
    assert range_violations({"colour.gcc_mean": None}) == {}  # missing or None is not a violation


@pytest.mark.parametrize("image_fixture", ["healthy_image", "stressed_image"])
def test_pipeline_features_stay_within_declared_bounds(image_fixture, request) -> None:
    image = request.getfixturevalue(image_fixture)
    features = Pipeline.default().analyze(image).plant_features
    assert range_violations(features.values) == {}


def test_leaf_features_stay_within_declared_bounds(healthy_image, leaf_segmenter) -> None:
    # The leaf provider populates plant.wilted_leaf_ratio, which is also declared-bounded.
    pipeline = Pipeline.default().with_region_provider(LeafInstanceRegionProvider(leaf_segmenter))
    features = pipeline.analyze(healthy_image).plant_features
    assert features.values["plant.wilted_leaf_ratio"] is not None
    assert range_violations(features.values) == {}

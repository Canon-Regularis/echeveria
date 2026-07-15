"""Contract invariants enforced by the core types (these underpin LSP guarantees)."""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.exceptions import ContractViolationError
from phytovision.types import FeatureVector, Region, RegionSet, StressAssessment


def test_region_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="empty mask"):
        Region(id=0, label="plant", mask=np.zeros((8, 8), dtype=bool), bbox=_dummy_bbox())


def test_region_rejects_non_boolean_mask() -> None:
    with pytest.raises(ContractViolationError, match="must be boolean"):
        Region(id=0, label="plant", mask=np.ones((8, 8), dtype=np.uint8), bbox=_dummy_bbox())


def test_regionset_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="at least one region"):
        RegionSet(regions=(), kind="plant", image_shape=(8, 8))


def test_stress_assessment_bounds() -> None:
    with pytest.raises(ValueError):
        StressAssessment(score=1.5, confidence=0.5, label="x", model_name="m")
    with pytest.raises(ValueError):
        StressAssessment(score=0.5, confidence=-0.1, label="x", model_name="m")


def test_feature_vector_merge_detects_collision() -> None:
    a = FeatureVector(region_id=0, values={"geometry.area_px": 1.0})
    b = FeatureVector(region_id=0, values={"geometry.area_px": 2.0})
    with pytest.raises(ValueError, match="collision"):
        a.merged_with(b)


def test_feature_vector_merge_rejects_cross_region() -> None:
    a = FeatureVector(region_id=0, values={"a": 1.0})
    b = FeatureVector(region_id=1, values={"b": 2.0})
    with pytest.raises(ValueError, match="across regions"):
        a.merged_with(b)


def _dummy_bbox():
    from phytovision.types import BBox

    return BBox(0, 0, 8, 8)

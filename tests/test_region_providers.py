from __future__ import annotations

import pytest

from phytovision.pipeline import Pipeline
from phytovision.regions.leaf_instance import LeafInstanceRegionProvider
from phytovision.regions.whole_plant import WholePlantRegionProvider

# The registry-driven substitutability contract lives in tests/contracts/test_provider_contract.py.
# This file keeps the provider-specific behaviour (region counts, downstream substitution).


def test_whole_plant_yields_single_region(healthy_image, plant_mask) -> None:
    result = WholePlantRegionProvider().regions(healthy_image, plant_mask)
    assert result.kind == "plant"
    assert len(result) == 1
    assert not result.is_per_leaf


def test_leaf_provider_yields_multiple_regions(healthy_image, plant_mask, leaf_segmenter) -> None:
    result = LeafInstanceRegionProvider(leaf_segmenter).regions(healthy_image, plant_mask)
    assert result.kind == "leaf"
    assert len(result) == 2
    assert result.is_per_leaf


def test_downstream_is_unchanged_by_substitution(healthy_image, leaf_segmenter) -> None:
    """The full pipeline runs and produces a valid report under BOTH providers."""
    base = Pipeline.default()
    pipelines = {
        "whole": base,
        "leaf": base.with_region_provider(LeafInstanceRegionProvider(leaf_segmenter)),
    }

    for name, pipeline in pipelines.items():
        report = pipeline.analyze(healthy_image)
        # Downstream contracts still hold regardless of region granularity.
        assert 0.0 <= report.stress.score <= 1.0
        assert 0.0 <= report.stress.confidence <= 1.0
        assert report.explanation.reasons  # the model stays explainable

        instance_only = ("plant.leaf_count", "plant.wilted_leaf_ratio")
        if report.regions.is_per_leaf:
            for key in instance_only:
                assert report.plant_features.values[key] is not None, (name, key)
            assert report.plant_features.values["plant.leaf_count"] == 2
        else:
            for key in instance_only:
                assert report.plant_features.values[key] is None, (name, key)


def test_aggregator_matches_region_count(healthy_image, leaf_segmenter) -> None:
    whole = Pipeline.default().analyze(healthy_image)
    leaf = (
        Pipeline.default()
        .with_region_provider(LeafInstanceRegionProvider(leaf_segmenter))
        .analyze(healthy_image)
    )

    assert whole.plant_features.region_count == 1
    assert leaf.plant_features.region_count == 2
    # Total measured leaf area is comparable whether measured as 1 plant or its 2 halves.
    whole_area = whole.plant_features.values["plant.total_area_px"]
    leaf_area = leaf.plant_features.values["plant.total_area_px"]
    assert whole_area == pytest.approx(leaf_area, rel=0.02)

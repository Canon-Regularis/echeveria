"""The no-training watershed leaf segmenter and its integration into the leaf-instance provider."""

from __future__ import annotations

import numpy as np

from phytovision.pipeline import Pipeline
from phytovision.regions.leaf_instance import LeafInstanceRegionProvider
from phytovision.registries import LEAF_SEGMENTERS, REGION_PROVIDERS
from phytovision.segmentation.leaves.watershed import WatershedLeafSegmenter


def _disk(cy: int, cx: int, r: int, size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


_IMAGE = np.zeros((64, 64, 3), np.float32)


def test_single_lobe_is_not_over_split() -> None:
    leaves = WatershedLeafSegmenter().segment_leaves(_IMAGE, _disk(32, 32, 18))
    assert len(leaves) == 1


def test_two_lobes_split_into_two_leaves() -> None:
    peanut = _disk(32, 22, 14) | _disk(32, 42, 14)
    leaves = WatershedLeafSegmenter().segment_leaves(_IMAGE, peanut)
    assert len(leaves) == 2
    for leaf in leaves:
        assert leaf.dtype == np.bool_ and leaf.shape == peanut.shape and leaf.any()


def test_empty_mask_yields_no_leaves() -> None:
    assert WatershedLeafSegmenter().segment_leaves(_IMAGE, np.zeros((64, 64), bool)) == []


def test_tiny_leaves_are_filtered_and_fall_back_to_the_whole_plant() -> None:
    peanut = _disk(32, 22, 14) | _disk(32, 42, 14)
    # min_leaf_fraction=1.0 requires a leaf to be the entire plant to survive. Both watershed halves
    # are smaller, so all are filtered and the segmenter falls back to one whole-plant region rather
    # than returning an empty list.
    leaves = WatershedLeafSegmenter(min_leaf_fraction=1.0).segment_leaves(_IMAGE, peanut)
    assert len(leaves) == 1
    assert np.array_equal(leaves[0], peanut)


def test_watershed_is_the_default_leaf_segmenter() -> None:
    assert "watershed" in LEAF_SEGMENTERS.names()
    provider = REGION_PROVIDERS.create("leaf-instance")
    assert isinstance(provider, LeafInstanceRegionProvider)
    assert isinstance(provider.segmenter, WatershedLeafSegmenter)


def test_leaf_provider_populates_per_leaf_traits(healthy_image) -> None:
    # Two lobes so the watershed actually splits; per-leaf traits then become defined.
    plant_mask = _disk(64, 44, 28, size=128) | _disk(64, 84, 28, size=128)
    provider = LeafInstanceRegionProvider(WatershedLeafSegmenter())
    region_set = provider.regions(np.zeros((128, 128, 3), np.float32), plant_mask)
    assert region_set.is_per_leaf
    assert len(region_set) == 2


def test_config_selects_the_leaf_provider(healthy_image) -> None:
    pipeline = Pipeline.from_config(
        {"region_provider": {"name": "leaf-instance", "params": {"leaf_segmenter": "watershed"}}}
    )
    report = pipeline.analyze(healthy_image)
    # A round synthetic plant is one lobe, so leaf_count is 1 but the per-leaf slots are populated.
    assert report.plant_features.values["plant.leaf_count"] is not None

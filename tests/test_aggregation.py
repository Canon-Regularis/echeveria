"""Plant-level aggregation: reduction policy, instance-only fields, wilted ratio."""

from __future__ import annotations

import numpy as np

from phytovision.phenotyping.aggregation.plant_level import PlantLevelAggregator
from phytovision.regions.base import region_from_mask
from phytovision.types import FeatureVector, RegionSet


def _leaf_regions(masks):
    regions = tuple(region_from_mask(i, "leaf", m) for i, m in enumerate(masks))
    return RegionSet(regions=regions, kind="leaf", image_shape=masks[0].shape)


def test_reduction_policy_sum_vs_weighted_mean() -> None:
    big = np.zeros((10, 10), dtype=bool)
    big[:4, :] = True  # area 40
    small = np.zeros((10, 10), dtype=bool)
    small[:2, :] = True  # area 20
    regions = _leaf_regions([big, small])
    features = [
        FeatureVector(0, {"geometry.area_px": 40.0}),
        FeatureVector(1, {"geometry.area_px": 20.0}),
    ]
    agg = PlantLevelAggregator()

    summed = agg.aggregate(regions, features, reduction_policy={"geometry.area_px": "sum"})
    assert summed.values["geometry.area_px"] == 60.0

    meaned = agg.aggregate(regions, features, reduction_policy={})
    # area-weighted mean of [40, 20] weighted by [40, 20] = 2000 / 60
    assert meaned.values["geometry.area_px"] == np.average([40.0, 20.0], weights=[40, 20])


def test_wilted_ratio_uses_configured_keys() -> None:
    mask = np.ones((4, 4), dtype=bool)
    regions = _leaf_regions([mask, mask])
    features = [
        FeatureVector(0, {"colour.yellow_fraction": 0.2, "colour.brown_fraction": 0.2}),  # 0.4
        FeatureVector(1, {"colour.yellow_fraction": 0.05, "colour.brown_fraction": 0.05}),  # 0.1
    ]
    out = PlantLevelAggregator(wilt_senescence_threshold=0.30).aggregate(regions, features)
    assert out.values["plant.wilted_leaf_ratio"] == 0.5
    assert out.values["plant.leaf_count"] == 2.0


def test_wilted_ratio_none_without_senescence_keys() -> None:
    mask = np.ones((4, 4), dtype=bool)
    regions = _leaf_regions([mask])
    out = PlantLevelAggregator().aggregate(regions, [FeatureVector(0, {"geometry.area_px": 16.0})])
    assert out.values["plant.wilted_leaf_ratio"] is None


def test_plant_kind_nulls_instance_only_fields(plant_region) -> None:
    regions = RegionSet(regions=(plant_region,), kind="plant", image_shape=plant_region.mask.shape)
    out = PlantLevelAggregator().aggregate(regions, [FeatureVector(0, {"colour.gcc_mean": 0.4})])
    assert out.values["plant.leaf_count"] is None
    assert out.values["plant.wilted_leaf_ratio"] is None
    assert out.values["plant.region_count"] == 1.0

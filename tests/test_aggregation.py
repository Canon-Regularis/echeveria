"""Plant-level aggregation: reduction policy, instance-only fields, wilted ratio."""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.phenotyping.aggregation.plant_level import PlantLevelAggregator
from phytovision.phenotyping.geometry import GeometryFeatures
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


def test_circular_reduction_stays_below_one_at_the_seam() -> None:
    # Two leaves whose hues straddle the wraparound average to a vector a hair below the seam; the
    # modulo then rounds up to exactly 1.0, outside the [0, 1) hue range and the opposite end of the
    # linear feature range from the ~0.0 it should read as.
    mask = np.ones((4, 4), dtype=bool)
    regions = _leaf_regions([mask, mask])
    features = [
        FeatureVector(0, {"colour.hue_mean": 0.02}),
        FeatureVector(1, {"colour.hue_mean": 0.98}),
    ]
    out = PlantLevelAggregator().aggregate(
        regions, features, reduction_policy={"colour.hue_mean": "circular"}
    )
    hue = out.values["colour.hue_mean"]
    assert 0.0 <= hue < 1.0
    assert min(hue, 1.0 - hue) < 0.01  # stays on the red seam, not the opposite hue


def test_orientation_reduces_as_an_axial_angle() -> None:
    # orientation is an undirected axis in [-pi/2, pi/2]: two near-horizontal leaves tilted to
    # opposite sides of the seam average linearly to 0.0 (vertical, the wrong axis). The axial
    # reduction on the doubled angle keeps the aggregate on the horizontal axis near +-pi/2.
    assert GeometryFeatures().reduction_policy()["geometry.orientation"] == "axial"

    mask = np.ones((4, 4), dtype=bool)
    regions = _leaf_regions([mask, mask])
    features = [
        FeatureVector(0, {"geometry.orientation": 1.4237}),
        FeatureVector(1, {"geometry.orientation": -1.4237}),
    ]
    out = PlantLevelAggregator().aggregate(
        regions, features, reduction_policy={"geometry.orientation": "axial"}
    )
    assert abs(abs(out.values["geometry.orientation"]) - np.pi / 2) < 1e-9


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


def test_wilted_ratio_counts_a_leaf_at_the_exact_threshold() -> None:
    # A leaf whose senescence sum is exactly the threshold counts as wilted (the test is `>=`, not
    # `>`). The existing wilted-ratio test uses 0.4 and 0.1 and never lands on the boundary, so a
    # `>=` -> `>` mutation survives it; here a leaf at exactly 0.30 makes the ratio 0.5, not 0.0.
    mask = np.ones((4, 4), dtype=bool)
    regions = _leaf_regions([mask, mask])
    features = [
        FeatureVector(0, {"colour.yellow_fraction": 0.30, "colour.brown_fraction": 0.0}),  # 0.30
        FeatureVector(1, {"colour.yellow_fraction": 0.10, "colour.brown_fraction": 0.0}),  # 0.10
    ]
    out = PlantLevelAggregator(wilt_senescence_threshold=0.30).aggregate(regions, features)
    assert out.values["plant.wilted_leaf_ratio"] == 0.5


def test_circular_and_axial_means_are_area_weighted() -> None:
    # The existing circular/axial tests use two equal masks, so the area weights never matter and
    # dropping them (np.average with weights=None) would pass. With a 90px leaf and a 10px leaf the
    # weighted mean leans toward the larger leaf's value, distinct from the unweighted midpoint.
    big = np.zeros((10, 10), dtype=bool)
    big[:9, :] = True  # area 90
    small = np.zeros((10, 10), dtype=bool)
    small[9:, :] = True  # area 10
    regions = _leaf_regions([big, small])
    agg = PlantLevelAggregator()

    circular = agg.aggregate(
        regions,
        [FeatureVector(0, {"colour.hue_mean": 0.10}), FeatureVector(1, {"colour.hue_mean": 0.90})],
        reduction_policy={"colour.hue_mean": "circular"},
    )
    assert circular.values["colour.hue_mean"] == pytest.approx(
        0.0838, abs=1e-3
    )  # not 0.0 unweighted

    axial = agg.aggregate(
        regions,
        [
            FeatureVector(0, {"geometry.orientation": 1.4}),
            FeatureVector(1, {"geometry.orientation": -1.4}),
        ],
        reduction_policy={"geometry.orientation": "axial"},
    )
    assert axial.values["geometry.orientation"] == pytest.approx(1.4322, abs=1e-3)  # not pi/2


def test_mean_region_area_is_total_over_region_count() -> None:
    # plant.mean_region_area is total area over region count; no existing test pins it, so a divisor
    # mutation (len - 1, union_area, image_area) would survive. Two leaves of 40 and 20 px give 30.
    big = np.zeros((10, 10), dtype=bool)
    big[:4, :] = True  # area 40
    small = np.zeros((10, 10), dtype=bool)
    small[:2, :] = True  # area 20
    regions = _leaf_regions([big, small])
    features = [
        FeatureVector(0, {"geometry.area_px": 40.0}),
        FeatureVector(1, {"geometry.area_px": 20.0}),
    ]
    out = PlantLevelAggregator().aggregate(regions, features)
    assert out.values["plant.total_area_px"] == 60.0
    assert out.values["plant.mean_region_area"] == pytest.approx(30.0)


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


def test_area_fraction_is_the_union_coverage_not_a_per_region_average() -> None:
    # area_fraction divides by the whole image, not the region, so at plant level it is the fraction
    # of the frame the plant occupies (the union coverage). Averaging it across k leaves reported
    # 1/k of the true area and contradicted area_px and canopy_coverage.
    from phytovision.phenotyping.geometry import GeometryFeatures

    first = np.zeros((100, 100), dtype=bool)
    first[10:33, 10:40] = True
    second = np.zeros((100, 100), dtype=bool)
    second[60:83, 60:90] = True  # disjoint from the first
    regions = _leaf_regions([first, second])
    extractor = GeometryFeatures()
    image = np.zeros((100, 100, 3), np.float32)
    features = [extractor.extract(image, region) for region in regions]

    values = (
        PlantLevelAggregator()
        .aggregate(regions, features, reduction_policy=extractor.reduction_policy())
        .values
    )
    assert values["geometry.area_fraction"] == pytest.approx(values["plant.canopy_coverage"])
    # disjoint leaves: the union is the sum of areas, so it also matches area_px / frame.
    assert values["geometry.area_fraction"] == pytest.approx(values["geometry.area_px"] / 10000.0)


def test_area_fraction_stays_bounded_when_masks_overlap() -> None:
    # Overlapping masks must not push the fraction past 1.0: it is the union, not the sum, matching
    # how canopy_coverage is defined.
    from phytovision.phenotyping.geometry import GeometryFeatures

    whole = np.zeros((50, 50), dtype=bool)
    whole[10:40, 10:40] = True
    regions = _leaf_regions([whole, whole, whole])  # three copies of the same region
    extractor = GeometryFeatures()
    image = np.zeros((50, 50, 3), np.float32)
    features = [extractor.extract(image, region) for region in regions]

    values = (
        PlantLevelAggregator()
        .aggregate(regions, features, reduction_policy=extractor.reduction_policy())
        .values
    )
    assert values["geometry.area_fraction"] <= 1.0
    assert values["geometry.area_fraction"] == pytest.approx(values["plant.canopy_coverage"])

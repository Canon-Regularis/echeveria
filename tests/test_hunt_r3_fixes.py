"""Regression tests for the round-3 bug-hunt findings.

Watershed leaves now tile the plant, the aggregator averages hue on the circle, a model forecaster's
current level is its last observation, the YOLO loader skips a non-finite class id, config reading
accepts a UTF-8 BOM, and the simulator's manifest label agrees with its stored score.
"""

from __future__ import annotations

import numpy as np

from phytovision.regions.base import region_from_mask
from phytovision.types import FeatureVector, RegionSet


def test_watershed_leaves_tile_the_plant_when_a_basin_is_dropped() -> None:
    from skimage.draw import disk

    from phytovision.segmentation.leaves.watershed import WatershedLeafSegmenter

    size = 128
    plant = np.zeros((size, size), dtype=bool)
    for cy, cx, r in [(64, 40, 22), (64, 88, 22), (30, 64, 8)]:  # two big lobes and a small one
        rr, cc = disk((cy, cx), r, shape=(size, size))
        plant[rr, cc] = True
    image = np.zeros((size, size, 3), dtype=np.float32)

    leaves = WatershedLeafSegmenter(min_leaf_fraction=0.10).segment_leaves(image, plant)
    union = np.logical_or.reduce(leaves)
    # The small lobe is below the 10% threshold; its pixels must flow into a kept leaf, not vanish,
    # so the union of the returned leaves still equals the plant foreground.
    assert int(union.sum()) == int(plant.sum())


def test_aggregator_averages_hue_on_the_circle() -> None:
    from phytovision.phenotyping.aggregation.plant_level import PlantLevelAggregator

    left = np.zeros((10, 10), dtype=bool)
    left[:, :5] = True
    right = np.zeros((10, 10), dtype=bool)
    right[:, 5:] = True
    regions = RegionSet(
        (region_from_mask(0, "leaf", left), region_from_mask(1, "leaf", right)), "leaf", (10, 10)
    )
    # Two equal-area leaves, both visually red but on either side of the hue wraparound.
    features = [
        FeatureVector(0, {"colour.hue_mean": 0.02}),
        FeatureVector(1, {"colour.hue_mean": 0.98}),
    ]
    plant = PlantLevelAggregator().aggregate(regions, features, {"colour.hue_mean": "circular"})
    hue = plant.values["colour.hue_mean"]
    assert hue is not None and (
        hue < 0.05 or hue > 0.95
    )  # near red, not the linear mean 0.5 (cyan)


def test_model_forecaster_current_level_is_the_last_observation() -> None:
    from phytovision.models.forecasting.base import Prediction, SeriesForecaster

    class _Flat(SeriesForecaster):
        name = "flat-test"

        def _predict(self, scores, steps):
            projected = {step: scores[-1] for step in steps}
            return Prediction(projected, dict(projected), dict(projected))

    scores = [0.30, 0.45, 0.60, 0.75, 0.90, 0.80, 0.70, 0.60, 0.50]  # rose then clearly declined
    forecast = _Flat().forecast(scores, (1, 3, 7))
    # The current level is the last observation (0.50), not the global linear fit (~0.72) a
    # non-linear model never used; a 0.72 would falsely read "stressed now".
    assert abs(forecast.current_level - 0.50) < 1e-9


def test_yolo_loader_skips_a_non_finite_class_id(tmp_path) -> None:
    from phytovision.datasets.yolo import YoloDetectionLoader

    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"not a real image, but a file")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("inf 0.5 0.5 0.2 0.3\n0 0.4 0.4 0.1 0.1\n", encoding="utf-8")

    samples = list(YoloDetectionLoader(str(images)))  # int(inf) must not escape as OverflowError
    assert len(samples) == 1


def test_read_config_accepts_a_utf8_bom(tmp_path) -> None:
    from phytovision.config import read_config

    config = tmp_path / "config.json"
    config.write_bytes(
        b"\xef\xbb\xbf" + b'{"model": "heuristic"}'
    )  # BOM-prefixed, as Notepad saves
    assert read_config(str(config)) == {"model": "heuristic"}


def test_simulate_manifest_label_agrees_with_its_stored_score(tmp_path) -> None:
    import csv

    from phytovision.models.base import bucket_label
    from phytovision.simulation import DryDownParams, simulate_cohort, write_manifest

    cohort = simulate_cohort(6, DryDownParams(n_steps=20), seed=3)
    manifest = write_manifest(cohort, tmp_path / "cohort.csv")
    for row in csv.DictReader(manifest.open()):
        # The label was bucketed from the same rounded score the row stores, so re-bucketing agrees.
        assert row["label"] == bucket_label(float(row["stress_score"]))

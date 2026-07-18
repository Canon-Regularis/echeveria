"""Regression tests for the intensive bug-hunt findings.

Two of these guard against regressions the earlier fixes introduced (a thin-region GLCM and a
degenerate forecast interval); the rest pin pre-existing defects the deep sweep surfaced: a producer
that ignored its coverage level, a loader missing an is_file guard, an off-by-one mask cleanup, and
a linear mean of a circular hue.
"""

from __future__ import annotations

import numpy as np

from phytovision.regions.base import region_from_mask


def test_glcm_thin_region_uses_only_valid_scan_directions() -> None:
    from phytovision.phenotyping.texture import TextureFeatures

    image = np.full((6, 20, 3), 0.5, dtype=np.float32)  # uniform grey, so texture is trivial
    mask = np.zeros((6, 20), dtype=bool)
    mask[2, 3:17] = True  # a single-row region: only the horizontal scan direction has pairs
    values = TextureFeatures().extract(image, region_from_mask(0, "plant", mask)).values
    # Averaging over the valid direction alone gives the true diagonal GLCM (homogeneity/energy 1),
    # not the ~0.25 the empty vertical directions used to drag the 4-angle mean down to.
    assert values["texture.glcm_homogeneity"] > 0.9
    assert values["texture.glcm_energy"] > 0.9


def test_degenerate_forecast_carries_no_interval() -> None:
    from phytovision.temporal.forecast import forecast_scores

    forecast = forecast_scores("p", [0.5], [1, 3, 7])
    assert forecast.lower == {} and forecast.upper == {}  # empty, matching the documented contract
    assert forecast.projected_scores  # the point projection is still present


def test_benchmark_threads_interval_level_into_the_producer() -> None:
    from phytovision.evaluation.benchmark import benchmark_forecasters
    from phytovision.temporal import FeatureHistory
    from phytovision.temporal.history import Observation

    history = FeatureHistory()
    for plant_id in ("a", "b", "c"):
        for i in range(6):
            history.add(
                Observation(
                    plant_id=plant_id, timestamp=f"2024-01-{i + 1:02d}", stress_score=0.2 + 0.1 * i
                )
            )
    narrow = benchmark_forecasters(
        history, ["linear-trend"], (1, 3), min_train=2, interval_level=0.5
    )
    wide = benchmark_forecasters(
        history, ["linear-trend"], (1, 3), min_train=2, interval_level=0.95
    )
    assert narrow.scores and wide.scores
    # The producer now builds intervals at the requested coverage, so 95% bands are wider than 50%;
    # the bug produced identical (always-90%) widths regardless of the requested level.
    assert sum(s.mean_width for s in wide.scores) > sum(s.mean_width for s in narrow.scores)


def test_yolo_loader_skips_directories_named_like_images(tmp_path) -> None:
    from phytovision.datasets.yolo import YoloDetectionLoader

    images = tmp_path / "images"
    images.mkdir()
    (images / "a.jpg").write_bytes(b"not a real image, but a file")
    (images / "crops.jpg").mkdir()  # a directory whose name ends in an image suffix

    paths = [sample.image_path for sample in YoloDetectionLoader(str(images))]
    assert any(path.endswith("a.jpg") for path in paths)
    assert not any(path.endswith("crops.jpg") for path in paths)


def test_clean_mask_keeps_a_region_exactly_at_min_size() -> None:
    from phytovision.segmentation.cleanup import clean_mask

    shape = (100, 100)  # min_object_fraction 0.002 -> min_size = 20
    mask = np.zeros(shape, dtype=bool)
    mask[0, :20] = True  # a component of exactly 20 pixels
    out = clean_mask(mask, shape, closing_radius=0)
    # The boundary size is kept now (dropped only below the fraction); the old max_size=min_size
    # deleted it, wiping a genuine 20-pixel plant to an empty mask.
    assert int(out.sum()) == 20


def test_circular_hue_mean_handles_the_wraparound() -> None:
    from phytovision.phenotyping.colour import circular_hue_mean

    red = np.array([0.02, 0.98, 0.01, 0.99])  # a coherent red split across the hue wraparound
    mean = circular_hue_mean(red)
    assert mean < 0.05 or mean > 0.95  # near 0 (red), not the linear mean ~0.5 (cyan)
    # a cluster that does not wrap still reads as its ordinary mean
    assert abs(circular_hue_mean(np.array([0.30, 0.32, 0.34])) - 0.32) < 0.01

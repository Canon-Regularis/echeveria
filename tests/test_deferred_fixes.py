"""Regression tests for the deferred bug-hunt findings.

The texture descriptors no longer leak region silhouette, the quality and preprocessing scalers are
dtype-aware, ``normalize01`` survives a degenerate range, and the survival leaderboard surfaces a
model it could not score instead of dropping it.
"""

from __future__ import annotations

import numpy as np

from phytovision.phenotyping.texture import TextureFeatures
from phytovision.regions.base import region_from_mask


def _region(mask: np.ndarray, region_id: int = 0):
    return region_from_mask(region_id=region_id, label="plant", mask=mask)


def test_glcm_does_not_leak_region_shape() -> None:
    # One textured field, two regions with identical interior texture but different silhouettes: a
    # solid square, and the same square with a quadrant carved out (concave, so half its bbox is
    # background). The GLCM used to spike on the concave one because the constant background fill
    # counted as co-occurrence pairs; foreground-only pairs make the descriptors track the texture.
    rng = np.random.default_rng(0)
    image = rng.random((60, 60, 3)).astype(np.float32)
    solid = np.zeros((60, 60), dtype=bool)
    solid[10:50, 10:50] = True
    concave = solid.copy()
    concave[10:30, 10:30] = False

    tex = TextureFeatures()
    a = tex.extract(image, _region(solid)).values
    b = tex.extract(image, _region(concave, region_id=1)).values

    assert abs(a["texture.glcm_energy"] - b["texture.glcm_energy"]) < 0.03
    assert abs(a["texture.glcm_homogeneity"] - b["texture.glcm_homogeneity"]) < 0.05


def test_edge_density_excludes_the_segmentation_boundary() -> None:
    # A uniform brighter square on a darker background has zero internal texture: all of its edge
    # signal is the outline. The interior-only average must read ~0, not the boundary step.
    image = np.full((40, 40, 3), 0.1, dtype=np.float32)
    image[8:32, 8:32] = 0.5
    mask = np.zeros((40, 40), dtype=bool)
    mask[8:32, 8:32] = True

    edge_density = TextureFeatures().extract(image, _region(mask)).values["texture.edge_density"]
    assert edge_density < 1e-6


def test_quality_scales_uint8_even_when_its_max_is_one() -> None:
    from phytovision.quality import _luminance

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[0, 0] = 1  # a sparse near-black frame whose maximum pixel value is 1
    # Before the fix the value-max heuristic left this un-scaled (luminance ~0.7); the dtype-aware
    # scaler divides by 255, so the near-black frame reads near-black.
    assert float(_luminance(image).max()) < 0.01


def test_preprocessor_scales_uint16_by_its_full_range() -> None:
    from phytovision.preprocessing.basic import ResizeNormalizePreprocessor

    image = np.full((16, 16, 3), 30000, dtype=np.uint16)
    out = ResizeNormalizePreprocessor().process(image)
    # 30000 / 65535 ~ 0.458; the old hard-coded /255 would saturate the whole frame to 1.0.
    assert 0.4 < float(out.max()) < 0.5


def test_normalize01_survives_a_degenerate_range() -> None:
    from phytovision._num import normalize01

    assert normalize01(0.5, 3.0, 3.0) == 0.0  # below the collapsed point reads 0, not a crash
    assert normalize01(3.0, 3.0, 3.0) == 1.0  # at or above it reads 1
    assert normalize01(5.0, 3.0, 3.0) == 1.0


def test_survival_benchmark_surfaces_a_model_it_cannot_score() -> None:
    from phytovision.evaluation.survival import (
        _covariate_model_names,
        benchmark_survival_models,
    )
    from phytovision.temporal import FeatureHistory
    from phytovision.temporal.history import Observation

    # Four plants, each with two observations that never cross the stressed cut: every fold has zero
    # events, so no model can be scored. Each must appear in `skipped`, not vanish from the board.
    history = FeatureHistory()
    for plant_id in ("A", "B", "C", "D"):
        history.add(Observation(plant_id=plant_id, timestamp="2024-01-01", stress_score=0.20))
        history.add(Observation(plant_id=plant_id, timestamp="2024-01-02", stress_score=0.25))

    board = benchmark_survival_models(history, folds=4)
    assert board.scores == ()
    assert set(board.skipped) == set(_covariate_model_names())

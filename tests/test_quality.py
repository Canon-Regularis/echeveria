"""Input and segmentation quality gates."""

from __future__ import annotations

import numpy as np

from phytovision.quality import assess_quality, laplacian_variance


def _blob(size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(1)
    img = np.ones((size, size, 3), np.float32) * 0.1
    yy, xx = np.mgrid[0:size, 0:size]
    img[(yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size * 0.3) ** 2] = (0.16, 0.55, 0.15)
    return np.clip(img + rng.normal(0.0, 0.02, img.shape).astype(np.float32), 0.0, 1.0)


def test_a_normal_plant_photo_is_usable() -> None:
    result = assess_quality(_blob(), foreground_fraction=0.3)
    assert result.usable
    assert result.flags == ()


def test_a_uniform_frame_is_flagged() -> None:
    grey = np.full((64, 64, 3), 0.4, np.float32)
    result = assess_quality(grey, foreground_fraction=0.5)
    assert not result.usable
    assert "uniform_image" in result.flags


def test_a_gradient_with_no_detail_is_flagged_blurry() -> None:
    gradient = np.linspace(0.1, 0.7, 64, dtype=np.float32)[None, :, None] * np.ones((64, 64, 3))
    result = assess_quality(gradient.astype(np.float32), foreground_fraction=0.3)
    assert "blurry" in result.flags
    assert "uniform_image" not in result.flags  # a gradient spans brightness, so it is not uniform


def test_a_full_frame_mask_is_flagged_as_failed_segmentation() -> None:
    result = assess_quality(_blob(), foreground_fraction=1.0)
    assert "full_frame_foreground" in result.flags


def test_a_tiny_mask_is_flagged() -> None:
    result = assess_quality(_blob(), foreground_fraction=0.001)
    assert "tiny_foreground" in result.flags


def test_laplacian_variance_falls_as_detail_falls() -> None:
    rng = np.random.default_rng(0)
    noisy = rng.random((32, 32))
    flat = np.full((32, 32), 0.5)
    assert laplacian_variance(noisy) > laplacian_variance(flat)
    assert laplacian_variance(flat) == 0.0


def test_as_dict_is_json_shaped() -> None:
    keys = set(assess_quality(_blob(), foreground_fraction=0.3).as_dict())
    assert keys == {
        "usable",
        "flags",
        "warnings",
        "blur_score",
        "foreground_fraction",
        "luminance_std",
    }

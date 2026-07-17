"""Occlusion saliency: hide each plant patch, rerun the pipeline, and localize the drivers."""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.exceptions import ContractViolationError
from phytovision.occlusion import occlusion_saliency
from phytovision.pipeline import Pipeline
from phytovision.visualize import render_occlusion_overlay

_SIZE = 96
_CENTER = _SIZE / 2
_RADIUS = 30


def _half_yellow_plant() -> np.ndarray:
    """A green disk whose left half is yellowed, so the stress driver has a known location.

    Yellow keeps a positive excess-green, so the exg-otsu segmenter still calls it plant, but it
    raises the yellow fraction the stress model penalizes. Occluding the yellow half should then
    lower the score (positive saliency), and occluding the green half should raise it (negative).
    """
    image = np.full((_SIZE, _SIZE, 3), 0.10, dtype=np.float32)
    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    disk = (yy - _CENTER) ** 2 + (xx - _CENTER) ** 2 <= _RADIUS**2
    image[disk] = np.array([0.15, 0.60, 0.15], dtype=np.float32)  # healthy green
    left = disk & (xx < _CENTER)
    image[left] = np.array([0.62, 0.60, 0.12], dtype=np.float32)  # yellowed
    return image


def test_saliency_shape_and_range() -> None:
    image = _half_yellow_plant()
    saliency = occlusion_saliency(image, Pipeline.default(), patch=24, stride=24)
    assert saliency.shape == (_SIZE, _SIZE)
    assert np.isfinite(saliency).all()
    assert saliency.min() >= -1.0 - 1e-9 and saliency.max() <= 1.0 + 1e-9


def test_saliency_localizes_the_yellow_driver() -> None:
    image = _half_yellow_plant()
    saliency = occlusion_saliency(image, Pipeline.default(), patch=24, stride=24)

    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    disk = (yy - _CENTER) ** 2 + (xx - _CENTER) ** 2 <= _RADIUS**2
    left = disk & (xx < _CENTER)
    right = disk & (xx >= _CENTER)

    # Hiding the yellow half lowers the score, so it is painted positive; hiding the green half
    # raises it, so it is painted negative. The driver half must outrank the healthy half.
    assert saliency[left].mean() > saliency[right].mean()
    assert saliency[left].mean() > 0.0


def test_background_patches_are_skipped() -> None:
    # The top-left corner never touches the disk, so no patch is occluded there: it reads as zero.
    saliency = occlusion_saliency(_half_yellow_plant(), Pipeline.default(), patch=24, stride=24)
    assert np.all(saliency[0:24, 0:24] == 0.0)


def test_saliency_is_deterministic() -> None:
    image = _half_yellow_plant()
    first = occlusion_saliency(image, Pipeline.default(), patch=24, stride=24)
    second = occlusion_saliency(image, Pipeline.default(), patch=24, stride=24)
    assert np.array_equal(first, second)


def test_non_positive_patch_or_stride_is_rejected() -> None:
    image = _half_yellow_plant()
    with pytest.raises(ContractViolationError):
        occlusion_saliency(image, Pipeline.default(), patch=0, stride=8)
    with pytest.raises(ContractViolationError):
        occlusion_saliency(image, Pipeline.default(), patch=16, stride=-1)


def test_uint8_input_is_accepted_and_localizes_the_driver() -> None:
    # A uint8 image is normalized to [0, 1] first, so the map still finds the yellow driver.
    as_uint8 = (_half_yellow_plant() * 255).astype(np.uint8)
    saliency = occlusion_saliency(as_uint8, Pipeline.default(), patch=24, stride=24)

    yy, xx = np.mgrid[0:_SIZE, 0:_SIZE]
    disk = (yy - _CENTER) ** 2 + (xx - _CENTER) ** 2 <= _RADIUS**2
    left, right = disk & (xx < _CENTER), disk & (xx >= _CENTER)
    assert np.isfinite(saliency).all()
    assert saliency[left].mean() > saliency[right].mean()


def test_all_plant_frame_falls_back_to_the_overall_mean() -> None:
    # A frame with no background exercises the fill fallback and must not raise or return NaNs.
    green = np.full((_SIZE, _SIZE, 3), 0.0, dtype=np.float32)
    green[..., 1] = 0.6  # a uniform green field the segmenter reads as all plant
    saliency = occlusion_saliency(green, Pipeline.default(), patch=32, stride=32)
    assert saliency.shape == (_SIZE, _SIZE)
    assert np.isfinite(saliency).all()


def test_non_rgb_image_is_rejected() -> None:
    flat = np.zeros((_SIZE, _SIZE), dtype=np.float32)
    with pytest.raises(ContractViolationError):
        occlusion_saliency(flat, Pipeline.default())


def test_overlay_matches_input_size_and_is_rgb() -> None:
    image = _half_yellow_plant()
    overlay = render_occlusion_overlay(image, Pipeline.default(), patch=24, stride=24)
    assert overlay.size == (_SIZE, _SIZE)  # PIL reports (width, height)
    assert overlay.mode == "RGB"

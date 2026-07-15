"""Preprocessing and segmentation stages."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from phytovision.exceptions import InvalidImageError
from phytovision.preprocessing.basic import ResizeNormalizePreprocessor
from phytovision.registries import SEGMENTERS
from phytovision.segmentation.cleanup import clean_mask
from phytovision.segmentation.plant.exg_threshold import ExGThresholdSegmenter
from phytovision.segmentation.plant.lab_threshold import LabChromaSegmenter


def test_preprocess_normalizes_and_resizes() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (2000, 1000, 3)).astype(np.uint8)
    out = ResizeNormalizePreprocessor(max_size=512).process(img)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert max(out.shape[:2]) == 512


def test_preprocess_rejects_non_rgb() -> None:
    with pytest.raises(InvalidImageError):
        ResizeNormalizePreprocessor().process(np.zeros((4, 4), dtype=np.uint8))


def test_segmenter_finds_foreground_blob(healthy_image) -> None:
    mask = ExGThresholdSegmenter().segment(healthy_image)
    assert mask.dtype == np.bool_
    assert mask.any() and not mask.all()


def test_segmenter_uniform_image_falls_back_and_warns(caplog) -> None:
    uniform = np.full((32, 32, 3), 0.4, dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        mask = ExGThresholdSegmenter().segment(uniform)
    assert mask.all()  # last-resort: whole frame is foreground
    assert "entire frame" in caplog.text


def test_lab_segmenter_finds_a_purple_plant() -> None:
    # A purple plant on neutral grey: Excess-Green struggles here, Lab chroma does not.
    size, radius = 96, 30
    img = np.full((size, size, 3), 0.45, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    blob = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius**2
    img[blob] = (0.50, 0.15, 0.55)

    mask = LabChromaSegmenter().segment(img)

    assert mask.dtype == np.bool_
    assert mask.any() and not mask.all()
    assert mask[size // 2, size // 2]  # plant centre is foreground


def test_lab_segmenter_is_registered() -> None:
    assert "lab-chroma" in SEGMENTERS.names()


def test_clean_mask_keep_largest_drops_the_smaller_blob() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:8, 2:8] = True  # small blob
    mask[15:35, 15:35] = True  # large blob
    kept = clean_mask(mask, (40, 40), min_object_fraction=0.0, closing_radius=0, keep_largest=True)
    assert kept[25, 25]
    assert not kept[4, 4]

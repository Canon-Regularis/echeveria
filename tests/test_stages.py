"""Preprocessing and segmentation stages."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from phytovision.exceptions import InvalidImageError
from phytovision.preprocessing.basic import ResizeNormalizePreprocessor
from phytovision.segmentation.plant.exg_threshold import ExGThresholdSegmenter


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

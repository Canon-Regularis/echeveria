"""The overlay renderer (F3)."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from phytovision.pipeline import Pipeline
from phytovision.visualize import render_overlay


def test_render_overlay_matches_input_size_and_is_rgb(healthy_image) -> None:
    report = Pipeline.default().analyze(healthy_image)
    img = (healthy_image * 255).astype(np.uint8)

    overlay = render_overlay(img, report)

    assert isinstance(overlay, PILImage.Image)
    assert overlay.mode == "RGB"
    assert overlay.size == (img.shape[1], img.shape[0])  # (width, height)


def test_render_overlay_resizes_mask_to_a_larger_image(healthy_image) -> None:
    # report mask is at 128px analysis resolution; render onto a 256px image to confirm scaling.
    report = Pipeline.default().analyze(healthy_image)
    big = np.zeros((256, 256, 3), dtype=np.uint8)
    overlay = render_overlay(big, report)
    assert overlay.size == (256, 256)

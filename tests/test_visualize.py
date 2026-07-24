"""The overlay renderer (F3)."""

from __future__ import annotations

import numpy as np
import pytest
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


def test_overlay_renders_the_photo_the_report_describes(healthy_image) -> None:
    # The renderer must resolve the image range the same way the preprocessor does. Reading a float
    # frame with a stray pixel just over 1.0 as already-8-bit skipped the *255 scaling and rendered
    # the photo solid black, so the overlay annotated an image the report never saw.
    stray = healthy_image.astype(np.float32).copy()
    stray[0, 0] = 1.2
    report = Pipeline.default().analyze(stray)

    rendered = np.asarray(render_overlay(stray, report))
    baseline = np.asarray(render_overlay(healthy_image.astype(np.float32), report))
    assert rendered.mean() > 20.0  # not a black frame
    assert rendered.mean() == pytest.approx(baseline.mean(), rel=0.1)


def test_overlay_does_not_saturate_a_16_bit_image(healthy_image) -> None:
    # A 16-bit frame was clipped against 255 and came out pure white; it is scaled by its own range.
    u16 = (healthy_image * 65535).astype(np.uint16)
    report = Pipeline.default().analyze(u16)
    rendered = np.asarray(render_overlay(u16, report))
    assert rendered.max() <= 255
    assert (rendered == 255).mean() < 0.5  # not a saturated white frame

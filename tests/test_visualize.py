"""The overlay renderer (F3)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image as PILImage

from phytovision.pipeline import Pipeline
from phytovision.visualize import _signed_overlay, render_overlay


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


def test_render_overlay_tints_stressed_red_healthy_green(healthy_image) -> None:
    # The stress tint is `_HEALTHY * (1 - score) + _STRESSED * score`; a high score must pull the
    # plant region toward red (more R, less G) and a low score toward green. Other overlay tests
    # pin only size, mode, and not-black, so swapping _HEALTHY and _STRESSED would still pass.
    report = Pipeline.default().analyze(healthy_image)
    mask = report.plant_mask  # 128px analysis mask over a 128px image: no resize
    base = (healthy_image * 255).astype(np.uint8)

    stressed = np.asarray(
        render_overlay(base, replace(report, stress=replace(report.stress, score=1.0)))
    ).astype(float)
    healthy = np.asarray(
        render_overlay(base, replace(report, stress=replace(report.stress, score=0.0)))
    ).astype(float)

    assert stressed[mask][:, 0].mean() > healthy[mask][:, 0].mean()  # stressed is redder
    assert stressed[mask][:, 1].mean() < healthy[mask][:, 1].mean()  # and less green


def test_signed_overlay_paints_positive_red_negative_green() -> None:
    # The shared saliency/occlusion overlay paints `_STRESSED * (saliency > 0) + _HEALTHY *
    # (saliency < 0)`: positive (score-raising) pixels red, negative green. Swapping the two colours
    # would pass every existing overlay test, which checks only size and mode.
    base = np.full((64, 64, 3), 120, dtype=np.uint8)
    saliency = np.zeros((64, 64), dtype=np.float32)
    saliency[:, :32] = 1.0  # left half raised the score
    saliency[:, 32:] = -1.0  # right half lowered it
    rendered = np.asarray(_signed_overlay(base, saliency, "x", 0.5)).astype(float)

    below_caption = rendered[20:]  # skip the top caption band
    left, right = below_caption[:, :32], below_caption[:, 32:]
    assert left[:, :, 0].mean() > left[:, :, 1].mean()  # positive: red
    assert right[:, :, 1].mean() > right[:, :, 0].mean()  # negative: green


def test_signed_overlay_opacity_carries_strength_not_tint() -> None:
    # Opacity carries the strength; the colour is the pure sign tint. Base 120, saliency 0.5,
    # alpha 0.5, red is 120*(1 - 0.5*0.5) + 0.5*0.5*210 = 142.5 -> 142 after the uint8 cast. Scaling
    # the tint by the magnitude too (the documented double-scaling bug) would give 116 instead.
    base = np.full((40, 40, 3), 120, dtype=np.uint8)
    saliency = np.full((40, 40), 0.5, dtype=np.float32)
    rendered = np.asarray(_signed_overlay(base, saliency, "x", 0.5))
    assert rendered[30, 20, 0] == 142  # not 116

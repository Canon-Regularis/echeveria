"""Spatial pigment saliency."""

from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

from phytovision.models.base import StressModel
from phytovision.pipeline import Pipeline
from phytovision.saliency import pigment_saliency
from phytovision.types import PlantFeatures, StressAssessment
from phytovision.visualize import render_saliency_overlay


class _NonContribModel(StressModel):
    """A stress model that cannot attribute its score, to exercise the graceful degrade."""

    name = "flat"

    def predict(self, features: PlantFeatures) -> StressAssessment:
        return StressAssessment(0.5, 0.5, "mild", "flat")


def _plant(colour: tuple[float, float, float], size: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    img = np.ones((size, size, 3), np.float32) * 0.1
    img[(yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= (size * 0.3) ** 2] = colour
    return img


def test_saliency_matches_the_mask_shape() -> None:
    img = _plant((0.62, 0.5, 0.12))
    pipeline = Pipeline.default()
    report = pipeline.analyze(img)
    assert pigment_saliency(img, report, pipeline.model).shape == report.plant_mask.shape


def test_a_yellowing_plant_reads_positive_inside_the_plant() -> None:
    img = _plant((0.62, 0.5, 0.12))  # orange-yellow, which raises stress
    pipeline = Pipeline.default()
    report = pipeline.analyze(img)
    saliency = pigment_saliency(img, report, pipeline.model)
    assert saliency.max() > 0.0
    assert saliency[~report.plant_mask].sum() == 0.0  # background stays zero


def test_a_healthy_green_plant_reads_negative() -> None:
    img = _plant((0.16, 0.55, 0.15))  # green, which lowers stress
    pipeline = Pipeline.default()
    report = pipeline.analyze(img)
    assert pigment_saliency(img, report, pipeline.model).min() < 0.0


def test_a_model_without_contributions_yields_a_flat_map() -> None:
    img = _plant((0.62, 0.5, 0.12))
    report = Pipeline.default().analyze(img)
    assert not pigment_saliency(img, report, _NonContribModel()).any()


def test_render_saliency_overlay_returns_an_rgb_image() -> None:
    img = _plant((0.62, 0.5, 0.12))
    pipeline = Pipeline.default()
    report = pipeline.analyze(img)
    out = render_saliency_overlay(img, report, pipeline.model)
    assert isinstance(out, PILImage.Image)
    assert out.size == (img.shape[1], img.shape[0])
    assert out.mode == "RGB"

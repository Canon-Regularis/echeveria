"""Render an analysis result onto the photo: plant outline, stress tint, and a caption.

A picture of what the model segmented and scored is the clearest output for an explainable tool, and
it makes segmentation failures (background captured, plant missed) obvious in a way a scalar score
does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from phytovision.models.base import StressModel
from phytovision.occlusion import occlusion_saliency
from phytovision.saliency import pigment_saliency
from phytovision.types import AnalysisReport, Image, Mask

if TYPE_CHECKING:
    from phytovision.pipeline import Pipeline

_HEALTHY = np.array([40, 170, 60], dtype=np.float32)  # green
_STRESSED = np.array([210, 50, 40], dtype=np.float32)  # red
_BOUNDARY = (255, 235, 59)  # yellow outline


def render_overlay(image: Image, report: AnalysisReport, alpha: float = 0.45) -> PILImage.Image:
    """Draw the plant outline and a stress-coloured tint over ``image``, with a caption.

    ``report.plant_mask`` is at the pipeline's internal (resized) resolution, so it is scaled
    back to the input image's size before compositing.
    """
    base = _to_uint8_rgb(image)
    height, width = base.shape[:2]
    mask = _resize_mask(report.plant_mask, (height, width))

    tint = _HEALTHY * (1.0 - report.stress.score) + _STRESSED * report.stress.score
    out = base.astype(np.float32)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * tint
    out[find_boundaries(mask, mode="outer")] = _BOUNDARY

    rendered = PILImage.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    _draw_caption(rendered, report)
    return rendered


def render_saliency_overlay(
    image: Image, report: AnalysisReport, model: StressModel, alpha: float = 0.5
) -> PILImage.Image:
    """Tint the photo by a pigment saliency map: red where colour pixels raised the score, green
    where they lowered it. It localizes colour drivers only, so treat it as an RGB proxy of the
    score's source."""
    saliency = pigment_saliency(image, report, model)
    return _signed_overlay(image, saliency, "PIGMENT SALIENCY (RGB proxy)", alpha)


def render_occlusion_overlay(
    image: Image,
    pipeline: Pipeline,
    alpha: float = 0.5,
    *,
    patch: int = 24,
    stride: int = 12,
) -> PILImage.Image:
    """Tint the photo by a model-agnostic occlusion map: red where hiding a patch lowered the score
    (the patch was raising it), green where hiding it raised the score. It reruns the pipeline once
    per patch, so it is far slower than the pigment overlay; treat it as an RGB proxy, not a
    measurement."""
    saliency = occlusion_saliency(image, pipeline, patch=patch, stride=stride)
    return _signed_overlay(image, saliency, "OCCLUSION SALIENCY (model-agnostic)", alpha)


def _signed_overlay(
    image: Image, saliency: np.ndarray, caption: str, alpha: float
) -> PILImage.Image:
    """Paint a signed map in ``[-1, 1]`` onto the photo: red for positive, green for negative, at
    a strength-scaled opacity, with a caption. Shared by the pigment and occlusion overlays."""
    base = _to_uint8_rgb(image).astype(np.float32)
    height, width = base.shape[:2]
    if saliency.shape != (height, width):
        saliency = resize(saliency, (height, width), order=1)

    # The colour is the pure sign tint and the opacity carries the strength. Scaling the tint by the
    # magnitude as well would apply strength twice, pulling mid-strength pixels toward grey rather
    # than a true (lighter) red or green.
    positive = (saliency > 0.0).astype(np.float32)[..., None]
    negative = (saliency < 0.0).astype(np.float32)[..., None]
    tint = _STRESSED * positive + _HEALTHY * negative
    strength = np.abs(saliency)[..., None]
    out = base * (1.0 - alpha * strength) + alpha * strength * tint

    rendered = PILImage.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered, "RGBA")
    draw.rectangle([(0, 0), (rendered.width, 18)], fill=(0, 0, 0, 170))
    draw.text((4, 3), caption, fill=(255, 255, 255))
    return rendered


def _to_uint8_rgb(image: Image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        scaled = arr * 255.0 if float(arr.max(initial=0.0)) <= 1.0 else arr
        arr = np.clip(scaled, 0, 255).astype(np.uint8)
    return arr[..., :3]


def _resize_mask(mask: Mask, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return resize(mask.astype(np.float32), shape, order=0, anti_aliasing=False) > 0.5


def _draw_caption(img: PILImage.Image, report: AnalysisReport) -> None:
    stress = report.stress
    lines = [f"{stress.label.upper()}  score={stress.score:.2f}  conf={stress.confidence:.2f}"]
    if report.explanation.reasons:
        reason = report.explanation.reasons[0]
        lines.append(f"[{reason.marker}] {reason.feature}")

    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([(0, 0), (img.width, 12 * len(lines) + 6)], fill=(0, 0, 0, 170))
    draw.multiline_text((4, 3), "\n".join(lines), fill=(255, 255, 255))

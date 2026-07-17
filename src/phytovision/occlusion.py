"""Model-agnostic occlusion saliency: which image patches move the stress score.

The pigment saliency in :mod:`phytovision.saliency` maps a colour model's own feature
contributions back onto pixels, so it paints colour drivers only, and only for a model that can
attribute its score. This is the complement. It treats the whole pipeline as a black box: it
occludes each patch of the plant in turn, reruns the analysis, and measures how far the score moves.
A patch whose removal lowers the score was raising it, so it is painted positive; a patch whose
removal raises the score was holding it down, so it is painted negative. It reruns the pipeline once
per patch, so it is far slower than the pigment map and lives behind a flag. Every value is an RGB
proxy of the score's source, never a measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from skimage.transform import resize

from phytovision.exceptions import ContractViolationError
from phytovision.types import Image

if TYPE_CHECKING:
    from phytovision.pipeline import Pipeline


def occlusion_saliency(
    image: Image,
    pipeline: Pipeline,
    *,
    patch: int = 24,
    stride: int = 12,
    fill: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """A signed per-pixel map at the input resolution: how much each plant patch raised the score.

    For every patch that overlaps the plant, the patch is painted a neutral fill, the pipeline
    reruns, and the patch is credited with ``baseline_score - occluded_score``: positive where
    hiding the patch lowered the score (it pushed toward stressed), negative where hiding it raised
    the score. Overlapping patches average, and the result is normalized to roughly ``[-1, 1]``.
    Patches that miss the plant are skipped, so background reads as zero.

    :param image: an ``H x W x 3`` uint8 or float RGB array.
    :param pipeline: the analysed pipeline; it is rerun once per plant patch.
    :param patch: side length in pixels of each occluded square.
    :param stride: step between patch origins; a stride below ``patch`` overlaps them for a smoother
        map at the cost of more pipeline runs.
    :param fill: the RGB colour, each channel in ``[0, 1]``, that replaces an occluded patch.
        Defaults to the mean colour of the image background (non-plant pixels), so an occluded patch
        reads as "this became background" rather than "a grey box appeared".
    :raises ContractViolationError: if ``image`` is not ``H x W x 3`` or ``patch``/``stride`` is not
        positive.
    """
    if patch <= 0 or stride <= 0:
        raise ContractViolationError(f"patch and stride must be positive, got {patch=}, {stride=}")

    rgb = _as_float_rgb(image)
    height, width = rgb.shape[:2]
    baseline = pipeline.analyze(rgb)
    base_score = baseline.stress.score
    plant = _resize_mask(baseline.plant_mask, (height, width))
    fill_rgb = (
        _fill_colour(rgb, plant) if fill is None else np.clip(np.asarray(fill, float), 0.0, 1.0)
    )

    total = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            bottom = min(top + patch, height)
            right = min(left + patch, width)
            if not plant[top:bottom, left:right].any():
                continue  # occluding pure background moves nothing, so skip the pipeline run
            occluded = rgb.copy()
            occluded[top:bottom, left:right] = fill_rgb
            score = pipeline.analyze(occluded).stress.score
            total[top:bottom, left:right] += base_score - score
            counts[top:bottom, left:right] += 1.0

    saliency = np.divide(total, counts, out=np.zeros_like(total), where=counts > 0.0)
    peak = float(np.abs(saliency).max())
    if peak > 0.0:
        saliency /= peak  # normalize to roughly [-1, 1] so the overlay ramp stays stable
    return saliency


def _fill_colour(rgb: np.ndarray, plant: np.ndarray) -> np.ndarray:
    """Mean background colour, so an occluded plant patch reads as background, not a grey box."""
    background = rgb[~plant]
    if background.size == 0:  # the whole frame is plant: fall back to the overall mean
        return rgb.reshape(-1, 3).mean(axis=0)
    return background.mean(axis=0)


def _as_float_rgb(image: Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ContractViolationError(
            f"occlusion needs an H x W x 3 RGB image, got shape {arr.shape}"
        )
    arr = arr[..., :3]
    if float(arr.max(initial=0.0)) > 1.0:  # accept uint8 or float; work in [0, 1]
        arr = arr / 255.0
    return arr


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    return resize(mask.astype(np.float32), shape, order=0, anti_aliasing=False) > 0.5

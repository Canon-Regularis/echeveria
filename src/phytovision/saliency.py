"""Spatial saliency: put the model's colour drivers back onto the pixels that produced them.

The stress model scores a whole-plant feature vector, so its explanation is feature-level. This maps
the signed contributions of the colour features (yellowing, browning, reddening, greenness) onto the
pixels of each class, giving a per-pixel picture of where the score came from. It localizes colour
drivers only: shape and texture drivers are not painted, so the map can diverge from the score when
those dominate. Every value is an RGB proxy.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2hsv
from skimage.transform import resize

from phytovision.models.base import ContributionModel, StressModel
from phytovision.phenotyping.colour import pixel_class_masks
from phytovision.types import AnalysisReport

# Colour class -> the feature whose contribution it carries. Green lowers stress; the rest raise it.
_LOCALIZABLE = {
    "yellow": "colour.yellow_fraction",
    "brown": "colour.brown_fraction",
    "red": "colour.red_fraction",
    "green": "colour.greenness_ratio",
}


def pigment_saliency(image: np.ndarray, report: AnalysisReport, model: StressModel) -> np.ndarray:
    """A signed per-pixel map, at the plant-mask resolution, of how colour pixels moved the score.

    Positive means the pixel pushed the score toward stressed; negative means toward healthy. A
    model that cannot attribute its score yields an all-zero map. The values are an RGB proxy.
    """
    mask = report.plant_mask
    saliency = np.zeros(mask.shape, dtype=np.float64)
    if not isinstance(model, ContributionModel):
        return saliency  # the model cannot attribute, so there is nothing to localize

    contributions = model.contributions(report.plant_features)
    small = _to_float_rgb(image, mask.shape)
    r, g, b = small[..., 0], small[..., 1], small[..., 2]
    hsv = rgb2hsv(small)
    classes = pixel_class_masks(r, g, b, hsv[..., 0], hsv[..., 1], hsv[..., 2])

    for class_name, feature_key in _LOCALIZABLE.items():
        contribution = contributions.get(feature_key)
        if not contribution:  # missing or exactly zero adds nothing
            continue
        saliency[classes[class_name] & mask] += contribution

    peak = float(np.abs(saliency).max())
    if peak > 0.0:
        saliency /= peak  # normalize to roughly [-1, 1] so the overlay ramp is stable
    return saliency


def _to_float_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if float(arr.max(initial=0.0)) > 1.0:  # accept uint8 or float; work in [0, 1]
        arr = arr / 255.0
    arr = arr[..., :3]
    if arr.shape[:2] != shape:
        arr = resize(arr, shape, order=1, anti_aliasing=True)
    return arr

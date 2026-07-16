"""Colour descriptors over a region's foreground pixels.

Colour is the primary visual signal of water stress (yellowing/browning, loss of greenness and
saturation), so these features feed the stress model heavily.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2hsv, rgb2lab

from phytovision._num import EPS
from phytovision.phenotyping.base import FeatureExtractor
from phytovision.types import Image, Region


def pixel_class_masks(
    r: np.ndarray, g: np.ndarray, b: np.ndarray, hue: np.ndarray, sat: np.ndarray, val: np.ndarray
) -> dict[str, np.ndarray]:
    """Boolean masks for the pixel-localizable colour classes, over whatever pixels are passed in.

    The same thresholds drive the colour fractions here and the saliency map, so the two agree.
    Yellow sits at hue 0.11 to 0.20; brown is dark, low-saturation orange; red covers the hue
    wraparound plus purple and magenta; green marks pixels greener than they are red or blue.
    """
    return {
        "yellow": (hue >= 0.10) & (hue <= 0.20) & (sat > 0.25) & (val > 0.25),
        "brown": (hue >= 0.03) & (hue <= 0.13) & (sat > 0.15) & (val < 0.55),
        "red": ((hue <= 0.05) | (hue >= 0.80)) & (sat > 0.25) & (val > 0.20),
        "green": g > np.maximum(r, b),
    }


class ColourFeatures(FeatureExtractor):
    namespace = "colour"

    def _compute(self, image: Image, region: Region) -> dict[str, float]:
        mask = region.mask
        rgb = image[mask]  # (N, 3) foreground pixels
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        total = r + g + b + EPS

        hsv = rgb2hsv(rgb)  # convert the foreground pixels only, not the whole frame
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        lab = rgb2lab(rgb)

        exg = 2.0 * g - r - b  # excess green
        gcc = g / total  # green chromatic coordinate

        classes = pixel_class_masks(r, g, b, hue, sat, val)
        n = float(rgb.shape[0])

        return {
            "mean_r": float(r.mean()),
            "mean_g": float(g.mean()),
            "mean_b": float(b.mean()),
            "exg_mean": float(exg.mean()),
            "gcc_mean": float(gcc.mean()),
            "greenness_ratio": float(classes["green"].mean()),
            "hue_mean": float(hue.mean()),
            "saturation_mean": float(sat.mean()),
            "saturation_std": float(sat.std()),
            "value_mean": float(val.mean()),
            "lab_l_mean": float(lab[:, 0].mean()),
            "lab_a_mean": float(lab[:, 1].mean()),  # +a = red, -a = green
            "lab_b_mean": float(lab[:, 2].mean()),  # +b = yellow, -b = blue
            "yellow_fraction": float(classes["yellow"].sum()) / (n + EPS),
            "brown_fraction": float(classes["brown"].sum()) / (n + EPS),
            "red_fraction": float(classes["red"].sum()) / (n + EPS),  # anthocyanin (pigment stress)
        }

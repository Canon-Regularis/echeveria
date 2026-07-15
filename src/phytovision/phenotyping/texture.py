"""Texture descriptors (GLCM, LBP, entropy, edge density) over a region.

Water-stressed leaves often show increased surface irregularity (wrinkling, speckling), which
raises texture entropy / contrast and lowers homogeneity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel

from phytovision.phenotyping.base import FeatureExtractor
from phytovision.types import Image, Region

_EPS = 1e-9
_GLCM_LEVELS = 32


@lru_cache(maxsize=2)
def _gray_maps(image_bytes: bytes, shape: tuple[int, ...], dtype: str) -> Any:
    image = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
    gray = rgb2gray(image)
    return gray, sobel(gray)


def _texture_maps(image: Image) -> Any:
    """Grayscale and Sobel edges, cached by image content so per-leaf regions reuse the work."""
    contiguous = np.ascontiguousarray(image)
    return _gray_maps(contiguous.tobytes(), contiguous.shape, str(contiguous.dtype))


class TextureFeatures(FeatureExtractor):
    namespace = "texture"

    def _compute(self, image: Image, region: Region) -> dict[str, float]:
        gray, edges = _texture_maps(image)
        mask = region.mask
        fg = gray[mask]

        # --- entropy + edge density over the foreground pixels directly ---
        entropy = _shannon_entropy(fg)
        edge_density = float(edges[mask].mean())

        # --- GLCM / LBP need a 2-D patch; use the bbox crop with background flattened ---
        bb = region.bbox
        gray_crop = gray[bb.min_row : bb.max_row, bb.min_col : bb.max_col]
        mask_crop = mask[bb.min_row : bb.max_row, bb.min_col : bb.max_col]
        patch = gray_crop.copy()
        patch[~mask_crop] = float(fg.mean())  # neutralize background so it adds no spurious texture

        glcm_stats = _glcm_stats(patch)
        lbp_stats = _lbp_stats(patch, mask_crop)

        return {
            "entropy": entropy,
            "edge_density": edge_density,
            **glcm_stats,
            **lbp_stats,
        }


def _shannon_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=64, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64)
    p /= p.sum() + _EPS
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _glcm_stats(patch: np.ndarray) -> dict[str, float]:
    q = np.clip(patch * (_GLCM_LEVELS - 1), 0, _GLCM_LEVELS - 1).astype(np.uint8)
    glcm = graycomatrix(
        q,
        distances=[1],
        angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=_GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
    }


def _lbp_stats(patch: np.ndarray, mask_crop: np.ndarray) -> dict[str, float]:
    n_points, radius = 8, 1
    patch_u8 = (np.clip(patch, 0.0, 1.0) * 255).astype(np.uint8)  # LBP wants integer dtype
    lbp = local_binary_pattern(patch_u8, n_points, radius, method="uniform")
    lbp_fg = lbp[mask_crop]
    if lbp_fg.size == 0:
        return {"lbp_uniformity": 0.0, "lbp_entropy": 0.0}
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp_fg, bins=n_bins, range=(0, n_bins))
    p = hist.astype(np.float64)
    p /= p.sum() + _EPS
    nz = p[p > 0]
    return {
        "lbp_uniformity": float((p**2).sum()),
        "lbp_entropy": float(-(nz * np.log2(nz)).sum()),
    }

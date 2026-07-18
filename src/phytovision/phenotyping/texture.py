"""Texture descriptors (GLCM, LBP, entropy, edge density) over a region.

Water-stressed leaves often show increased surface irregularity (wrinkling, speckling), which
raises texture entropy / contrast and lowers homogeneity.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import sobel

from phytovision._num import EPS
from phytovision.phenotyping.base import FeatureExtractor
from phytovision.types import Image, Region

_GLCM_LEVELS = 32


@lru_cache(maxsize=2)
def _gray_maps(
    image_bytes: bytes, shape: tuple[int, ...], dtype: str
) -> tuple[np.ndarray, np.ndarray]:
    image = np.frombuffer(image_bytes, dtype=dtype).reshape(shape)
    gray = rgb2gray(image)
    return gray, sobel(gray)


def _texture_maps(image: Image) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale and Sobel edges, cached by image content so per-leaf regions reuse the work."""
    contiguous = np.ascontiguousarray(image)
    return _gray_maps(contiguous.tobytes(), contiguous.shape, str(contiguous.dtype))


class TextureFeatures(FeatureExtractor):
    namespace = "texture"

    def _compute(self, image: Image, region: Region) -> dict[str, float]:
        gray, edges = _texture_maps(image)
        mask = region.mask
        fg = gray[mask]

        # --- entropy over the foreground pixels; edge density over the interior only ---
        entropy = _shannon_entropy(fg)
        edge_density = _interior_edge_density(edges, mask)

        # --- GLCM / LBP need a 2-D patch: crop to the region's bbox ---
        bb = region.bbox
        gray_crop = gray[bb.min_row : bb.max_row, bb.min_col : bb.max_col]
        mask_crop = mask[bb.min_row : bb.max_row, bb.min_col : bb.max_col]
        patch = gray_crop.copy()
        patch[~mask_crop] = float(fg.mean())  # a neutral fill for LBP, whose codes read the mask

        glcm_stats = _glcm_stats(gray_crop, mask_crop)
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
    p /= p.sum() + EPS
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _interior_edge_density(edges: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel magnitude over the interior foreground only.

    Foreground pixels on the region outline sit on the strong plant/background intensity step, so
    averaging over the whole mask would let the silhouette (a shape signal) dominate what should
    measure internal surface texture. Eroding by one pixel drops that boundary ring; a region too
    thin to have an interior keeps its full mask so the value stays defined.
    """
    interior = binary_erosion(mask)
    region = interior if interior.any() else mask
    return float(edges[region].mean())


def _glcm_stats(gray_crop: np.ndarray, mask_crop: np.ndarray) -> dict[str, float]:
    """GLCM texture over foreground-foreground pixel pairs only.

    Background is mapped to a sentinel grey level and every co-occurrence that touches it is dropped
    before the matrix is renormalized, so a concave silhouette (which leaves a lot of background
    inside the bbox) cannot inflate homogeneity/energy the way a constant background fill would.
    """
    quantized = np.clip(gray_crop * (_GLCM_LEVELS - 1), 0, _GLCM_LEVELS - 1).astype(np.uint8)
    sentinel = _GLCM_LEVELS  # one extra level, reserved for background
    labelled = np.where(mask_crop, quantized, sentinel).astype(np.uint8)
    glcm = graycomatrix(
        labelled,
        distances=[1],
        angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=_GLCM_LEVELS + 1,
        symmetric=True,
        normed=False,
    )
    kept = glcm[:_GLCM_LEVELS, :_GLCM_LEVELS, :, :].astype(np.float64)  # drop the sentinel row/col
    totals = kept.sum(axis=(0, 1), keepdims=True)
    kept /= np.where(totals > 0.0, totals, 1.0)  # renormalize per angle over the kept pairs
    valid = totals.reshape(-1) > 0.0  # angles that actually had a foreground-foreground pair

    # Average each property only over directions that had pairs. A thin (1-row/1-col) region leaves
    # some angles empty; folding their graycoprops defaults (contrast/homogeneity/energy 0, but an
    # empty matrix scores correlation 1) into the mean would bias the rotation-averaged descriptor.
    def average(prop: str, empty: float) -> float:
        values = graycoprops(kept, prop).reshape(-1)
        return float(values[valid].mean()) if valid.any() else empty

    return {
        "glcm_contrast": average("contrast", 0.0),
        "glcm_homogeneity": average("homogeneity", 1.0),
        "glcm_energy": average("energy", 1.0),
        "glcm_correlation": average("correlation", 1.0),
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
    p /= p.sum() + EPS
    nz = p[p > 0]
    return {
        "lbp_uniformity": float((p**2).sum()),
        "lbp_entropy": float(-(nz * np.log2(nz)).sum()),
    }

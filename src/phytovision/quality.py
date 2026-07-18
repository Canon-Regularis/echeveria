"""Input and segmentation quality checks that flag an unanalysable image before it is trusted.

The checks are advisory: they never change the stress score. They add machine-readable flags and
human-readable warnings so a caller sees when a score is unreliable. A blurry photo, a near-uniform
frame with no plant, or a mask covering almost none or almost all of the frame each raise a flag.
Every check reads pixels, so treat it as an RGB proxy.

The module depends only on numpy, so importing it stays cheap and pulls in nothing heavy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Rec. 709 luma weights, matching skimage's rgb2gray, so luminance tracks the rest of the code.
_LUMA = np.array([0.2125, 0.7154, 0.0721])


@dataclass(frozen=True, slots=True)
class QualityAssessment:
    """The reliability of one analysis: usable overall, plus the flags and metrics behind it."""

    usable: bool
    flags: tuple[str, ...]  # machine-readable codes, e.g. "blurry", "full_frame_foreground"
    warnings: tuple[str, ...]  # human-readable notes for each flag
    blur_score: float  # variance of the Laplacian; lower means less detail
    foreground_fraction: float  # share of the frame the segmenter marked as plant
    luminance_std: float  # spread of brightness; near zero means a near-uniform frame

    def as_dict(self) -> dict[str, object]:
        """A compact, JSON-serializable digest for a report summary."""
        return {
            "usable": self.usable,
            "flags": list(self.flags),
            "warnings": list(self.warnings),
            "blur_score": round(self.blur_score, 6),
            "foreground_fraction": round(self.foreground_fraction, 4),
            "luminance_std": round(self.luminance_std, 4),
        }


def _luminance(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):  # scale by the dtype's full range, not a value guess
        arr = arr.astype(np.float64) / float(np.iinfo(arr.dtype).max)
    else:  # a float image is already in [0, 1]; a near-black one is not mistaken for uint8
        arr = arr.astype(np.float64)
    return arr[..., :3] @ _LUMA


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of a 4-neighbour Laplacian: a sharpness proxy that falls as an image blurs."""
    if min(gray.shape) < 3:
        return 0.0
    lap = (
        4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    return float(lap.var())


def assess_quality(
    image: np.ndarray,
    foreground_fraction: float,
    *,
    blur_min: float = 1e-4,
    uniform_std_min: float = 0.02,
    min_coverage: float = 0.005,
    max_coverage: float = 0.98,
) -> QualityAssessment:
    """Judge whether an analysis is reliable, from the input image and the plant coverage.

    :param foreground_fraction: share of the frame segmented as plant (``plant.canopy_coverage``).
    :param blur_min: Laplacian variance below this flags ``blurry``.
    :param uniform_std_min: luminance std below this flags ``uniform_image``.
    :param min_coverage: coverage below this flags ``tiny_foreground``.
    :param max_coverage: coverage above this flags ``full_frame_foreground``.
    """
    gray = _luminance(image)
    blur = laplacian_variance(gray)
    lum_std = float(gray.std())

    flags: list[str] = []
    warnings: list[str] = []
    if lum_std < uniform_std_min:
        flags.append("uniform_image")
        warnings.append("image is nearly uniform, so it may not contain a plant")
    if blur < blur_min:
        flags.append("blurry")
        warnings.append("image detail is very low, so the score may be unreliable")
    if foreground_fraction < min_coverage:
        flags.append("tiny_foreground")
        warnings.append(
            "the plant covers very little of the frame, so segmentation may have failed"
        )
    elif foreground_fraction > max_coverage:
        flags.append("full_frame_foreground")
        warnings.append("the plant fills the frame, so segmentation may have failed")

    return QualityAssessment(
        usable=not flags,
        flags=tuple(flags),
        warnings=tuple(warnings),
        blur_score=blur,
        foreground_fraction=float(foreground_fraction),
        luminance_std=lum_std,
    )

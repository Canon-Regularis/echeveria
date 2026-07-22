"""Shared input validation, so the RGB contract is enforced in exactly one place."""

from __future__ import annotations

import numpy as np

from phytovision.exceptions import InvalidImageError


def validate_rgb_image(image: object) -> None:
    """Raise :class:`InvalidImageError` unless ``image`` is a non-empty ``H x W x 3`` ndarray.

    Called once at the pipeline entry point; individual stages rely on this rather than
    re-implementing the check, so the input contract cannot drift between stages.
    """
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"expected a numpy ndarray image, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidImageError(f"expected an H x W x 3 RGB image, got shape {image.shape}")
    if image.size == 0:
        raise InvalidImageError("image is empty")
    # A NaN/inf pixel defeats the preprocessor's max-based range check and the quality thresholds,
    # silently corrupting the analysis, so it is rejected loudly here rather than propagated. A
    # non-numeric dtype (object, string) cannot be finite-checked at all: numpy raises a TypeError,
    # re-raised as an InvalidImageError so the entry-point contract stays a single exception type.
    try:
        finite = bool(np.isfinite(image).all())
    except TypeError as exc:
        raise InvalidImageError(f"image has a non-numeric dtype: {image.dtype}") from exc
    if not finite:
        raise InvalidImageError("image contains non-finite pixels (NaN or inf)")

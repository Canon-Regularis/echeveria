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

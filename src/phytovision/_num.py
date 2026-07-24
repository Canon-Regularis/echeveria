"""Small numeric helpers shared across the package.

This module imports nothing from the package, so any low-level module can depend on it without a
cycle. Keep it tiny: only genuinely cross-cutting helpers belong here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# A small positive constant used to keep divisions away from zero.
EPS = 1e-9

# A float image is taken as already normalized unless its maximum clears this cut, which reads as an
# 8-bit range. Above 1.0 rather than at it, so a stray pixel a hair over one does not darken a whole
# normalized frame by 255; see to_unit_rgb.
_EIGHT_BIT_CUT = 1.5


def to_unit_rgb(image: np.ndarray) -> np.ndarray:
    """An ``H x W x 3`` float image in [0, 1], however the caller expressed its range.

    One rule for the whole package, so the pipeline, the quality checks, and the renderers cannot
    disagree about whether an array is 8-bit or already normalized: an integer image is scaled by
    its own dtype range (so 16-bit input works, not just uint8), and a float image is taken as
    [0, 1] unless it is clearly an 8-bit range. Two converters drifting apart on that question meant
    an image the pipeline scored correctly could be rendered or explained as a black frame.
    """
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):  # scale by the dtype's range, not a hard 255
        scaled = arr.astype(np.float64) / float(np.iinfo(arr.dtype).max)
    else:
        scaled = arr.astype(np.float64)
        if float(scaled.max(initial=0.0)) > _EIGHT_BIT_CUT:
            scaled = scaled / 255.0
    return np.clip(scaled[..., :3], 0.0, 1.0)


def clip01(value: float) -> float:
    """Clamp a value into the closed unit interval [0, 1]."""
    return min(1.0, max(0.0, value))


def normalize01(value: float, lo: float, hi: float) -> float:
    """Scale ``value`` from the range [lo, hi] into [0, 1], clamped at both ends.

    A degenerate range (``hi <= lo``) has no interior to scale into, so it collapses to the clamp:
    below ``lo`` reads 0, at or above it reads 1, rather than dividing by zero.
    """
    span = hi - lo
    if span <= 0.0:
        return 0.0 if value < lo else 1.0
    return clip01((value - lo) / span)


def as_float(value: object, default: float) -> float:
    """Coerce an optional feature value to a float, using ``default`` when the value is None."""
    return default if value is None else float(value)  # type: ignore[arg-type]


def feature_value(values: Mapping[str, object], key: str, default: float) -> float:
    """Read ``key`` from a feature mapping as a float, using ``default`` when it is missing."""
    return as_float(values.get(key), default)

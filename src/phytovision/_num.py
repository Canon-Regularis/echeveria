"""Small numeric helpers shared across the package.

This module imports nothing from the package, so any low-level module can depend on it without a
cycle. Keep it tiny: only genuinely cross-cutting helpers belong here.
"""

from __future__ import annotations

from collections.abc import Mapping

# A small positive constant used to keep divisions away from zero.
EPS = 1e-9


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

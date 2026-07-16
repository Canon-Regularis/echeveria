"""Small numeric helpers shared across the package.

This module imports nothing from the package, so any low-level module can depend on it without a
cycle. Keep it tiny: only genuinely cross-cutting helpers belong here.
"""

from __future__ import annotations


def clip01(value: float) -> float:
    """Clamp a value into the closed unit interval [0, 1]."""
    return min(1.0, max(0.0, value))

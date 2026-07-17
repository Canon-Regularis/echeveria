"""The Streamlit dashboard: a basic, sleek water-stress terminal (needs the ``dashboard`` extra).

The package keeps the UI in focused modules: the pure data-shaping helpers in ``helpers``
(importable and testable with the base dependencies), the theme in ``theme``, one module per tab,
and the Streamlit entry point in ``app``. The pure helpers are re-exported here, so ``from
phytovision.dashboard import forecast_points`` and the like keep working.
"""

from __future__ import annotations

from phytovision.dashboard.app import render
from phytovision.dashboard.helpers import (
    contribution_series,
    decode_image,
    disease_series,
    drought_markers,
    forecast_band,
    forecast_points,
    observation_table,
    plant_survival_metrics,
    quality_banner,
    reason_rows,
    survival_curve_points,
    timing_rows,
)

__all__ = [
    "contribution_series",
    "decode_image",
    "disease_series",
    "drought_markers",
    "forecast_band",
    "forecast_points",
    "observation_table",
    "plant_survival_metrics",
    "quality_banner",
    "reason_rows",
    "render",
    "survival_curve_points",
    "timing_rows",
]

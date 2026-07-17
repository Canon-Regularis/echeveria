"""Turn a dataset of timestamped, plant-tagged images into a ``FeatureHistory``.

A sample needs both a ``plant_id`` and a ``timestamp`` to join a time series, so samples missing
either are skipped with a warning, as ``analyze_dataset`` skips unreadable images. This bridges the
dataset loaders (which carry plant_id/timestamp) to the temporal store. The batch feature export
does not need those two fields, so it does not build this; temporal does.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from phytovision.datasets.base import DatasetLoader
from phytovision.exceptions import PhytoVisionError
from phytovision.temporal.early_warning import EarlyWarning, pigment_early_warning
from phytovision.temporal.forecast import DEFAULT_HORIZONS, Forecast, stress_forecast
from phytovision.temporal.history import FeatureHistory
from phytovision.temporal.trend import StressTrend, stress_trend

if TYPE_CHECKING:
    # Imported for typing only. Keeping it out of runtime lets the temporal package be imported
    # (e.g. by the leaf-death forecaster via the registry) without dragging in the pipeline module.
    from phytovision.models.base import TrajectoryForecaster
    from phytovision.pipeline import Pipeline

_log = logging.getLogger(__name__)


def build_history(pipeline: Pipeline, loader: DatasetLoader) -> FeatureHistory:
    """Analyze every plant-tagged, timestamped sample in ``loader`` into a ``FeatureHistory``."""
    history = FeatureHistory()
    for sample in loader:
        if not sample.plant_id or not sample.timestamp:
            _log.warning("skipping %s: needs both plant_id and timestamp", sample.image_path)
            continue
        try:
            report = pipeline.analyze(sample.image_path)
        except (FileNotFoundError, PhytoVisionError) as exc:
            _log.warning("skipping %s: %s", sample.image_path, exc)
            continue
        history.record(sample.plant_id, sample.timestamp, report)
    return history


def plant_trends(history: FeatureHistory) -> dict[str, StressTrend]:
    """Fit a stress trend per plant over its time-ordered observations."""
    return {
        plant_id: stress_trend(plant_id, history.series_for(plant_id))
        for plant_id in history.plant_ids
    }


def plant_early_warnings(history: FeatureHistory) -> dict[str, EarlyWarning]:
    """Compute the pigment-before-collapse early warning per plant."""
    return {
        plant_id: pigment_early_warning(plant_id, history.series_for(plant_id))
        for plant_id in history.plant_ids
    }


def plant_forecasts(
    history: FeatureHistory,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    forecaster: TrajectoryForecaster | None = None,
) -> dict[str, Forecast]:
    """Project each plant's stress trajectory forward to the given horizons.

    With no ``forecaster`` this is the linear-trend default. Pass any registered
    ``TrajectoryForecaster`` to project with prediction intervals from a richer model instead.
    """
    return {
        plant_id: _forecast_plant(plant_id, history, horizons, forecaster)
        for plant_id in history.plant_ids
    }


def _forecast_plant(
    plant_id: str,
    history: FeatureHistory,
    horizons: Sequence[int],
    forecaster: TrajectoryForecaster | None,
) -> Forecast:
    series = history.series_for(plant_id)
    if forecaster is None:
        return stress_forecast(plant_id, series, horizons)
    scores = [observation.stress_score for observation in series]
    return forecaster.forecast(scores, horizons, plant_id)

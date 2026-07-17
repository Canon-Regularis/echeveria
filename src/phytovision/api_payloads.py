"""Shape a temporal analysis into the JSON the ``/trend`` endpoint returns.

This is the serialization behind the API, kept apart from the HTTP wiring so it carries no web
dependency and can be built and tested on its own. It turns a ``FeatureHistory`` into per-plant
trends, series, early warnings, forecasts, and an optional survival block, each labelled as an
RGB-proxy estimate rather than a measurement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from phytovision.models.survival import fit_cohort_survival
from phytovision.temporal import (
    FeatureHistory,
    plant_early_warnings,
    plant_forecasts,
    plant_trends,
)

if TYPE_CHECKING:
    from phytovision.models.base import TrajectoryForecaster
    from phytovision.models.survival import SurvivalFit

_DISCLAIMER = (
    "early_warning and forecast are RGB-proxy extrapolations, not predictions; the "
    "prediction interval is an uncertainty estimate, not a measurement"
)
_SURVIVAL_DISCLAIMER_SUFFIX = (
    "; the survival estimate is a synthetic-trained RGB-proxy indication of time-to-wilt, not a "
    "validated prognosis"
)


def trend_payload(
    history: FeatureHistory,
    forecaster: TrajectoryForecaster | None = None,
    survival_model: str | None = "weibull-aft",
) -> dict[str, object]:
    """Serialize per-plant trends, series, the early warning, the forecast, and survival to JSON."""
    warnings = plant_early_warnings(history)
    forecasts = plant_forecasts(history, forecaster=forecaster)
    survival_fit, survival_note = _survival_fit(history, survival_model)
    per_plant = survival_fit.per_plant if survival_fit is not None else {}
    plants: dict[str, object] = {}
    for plant_id, trend in plant_trends(history).items():
        warning = warnings[plant_id]
        forecast = forecasts[plant_id]
        entry: dict[str, object] = {
            "direction": trend.direction,
            "slope": round(trend.slope, 6),
            "n": trend.n,
            "start_score": round(trend.start_score, 4),
            "end_score": round(trend.end_score, 4),
            "early_warning": {
                "flagged": warning.flagged,
                "pigment_slope": round(warning.pigment_slope, 6),
                "note": warning.note,
            },
            "forecast": {
                "method": forecast.method,
                "projected_scores": {
                    str(horizon): round(score, 4)
                    for horizon, score in forecast.projected_scores.items()
                },
                "lower": {str(h): round(v, 4) for h, v in forecast.lower.items()},
                "upper": {str(h): round(v, 4) for h, v in forecast.upper.items()},
                "interval_level": forecast.interval_level,
                "steps_to_stressed": forecast.steps_to_stressed,
                "confidence": round(forecast.confidence, 4),
            },
            "series": [
                {"timestamp": obs.timestamp, "score": round(obs.stress_score, 4)}
                for obs in history.series_for(plant_id)
            ],
        }
        if plant_id in per_plant:
            plant_survival = per_plant[plant_id]
            entry["survival"] = {
                "basis": plant_survival.basis,
                "median": plant_survival.median,
                "lower": plant_survival.lower,
                "upper": plant_survival.upper,
            }
        plants[plant_id] = entry

    payload: dict[str, object] = {
        "plants": plants,
        "survival": _survival_summary(survival_fit),
        "disclaimer": _DISCLAIMER + _SURVIVAL_DISCLAIMER_SUFFIX if survival_fit else _DISCLAIMER,
    }
    if survival_note is not None:
        payload["survival_note"] = survival_note
    return payload


def _survival_fit(
    history: FeatureHistory, survival_model: str | None
) -> tuple[SurvivalFit | None, str | None]:
    """Fit the cohort survival, or degrade to no survival when the stats extra is absent."""
    if not survival_model:
        return None, None
    try:
        return fit_cohort_survival(history, survival_model), None
    except ImportError as exc:  # survival is additive: keep the forecast, note the omission
        return None, str(exc)


def _survival_summary(fit: SurvivalFit | None) -> dict[str, object] | None:
    if fit is None:
        return None
    return {
        "model": fit.model_name,
        "cohort_median": fit.cohort_median,
        "cohort_median_ci": list(fit.cohort_median_ci),
        "concordance_index": fit.concordance_index,
        "concordance_note": "in-sample, optimistic; see the survival benchmark for held-out scores",
        "curve": [
            {"t": t, "survival": s, "lower": lo, "upper": hi}
            for t, s, lo, hi in zip(
                fit.curve.times, fit.curve.survival, fit.curve.lower, fit.curve.upper, strict=True
            )
        ],
        "disclaimer": fit.disclaimer,
    }

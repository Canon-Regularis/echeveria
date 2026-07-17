"""Pure data-shaping helpers behind the dashboard.

Each function turns a report, a forecast, or a survival fit into the rows and series the UI draws.
None of them touch Streamlit or plotly, so they import and test with the base dependencies alone.
"""

from __future__ import annotations

from collections.abc import Sequence

from phytovision.io import decode_rgb_bytes
from phytovision.models.survival import SurvivalFit
from phytovision.temporal import Forecast, Observation
from phytovision.types import AnalysisReport, Image


def decode_image(data: bytes) -> Image:
    """Decode uploaded bytes into an RGB array, raising a clean domain error on junk input."""
    return decode_rgb_bytes(data)


def reason_rows(report: AnalysisReport) -> list[dict[str, object]]:
    """Flatten the explanation into display rows, strongest driver first."""
    return [
        {
            "feature": reason.feature,
            "value": round(reason.value, 4),
            "effect on stress": reason.direction,
            "contribution": round(reason.contribution, 4),
            "why": reason.description,
        }
        for reason in report.explanation.reasons
    ]


def contribution_series(report: AnalysisReport) -> tuple[list[str], list[float]]:
    """Feature names and signed contributions for a bar chart, largest magnitude first."""
    ranked = sorted(
        report.explanation.reasons, key=lambda reason: abs(reason.contribution), reverse=True
    )
    return [reason.feature for reason in ranked], [reason.contribution for reason in ranked]


def disease_series(report: AnalysisReport) -> tuple[list[str], list[float]]:
    """Disease-class labels and probabilities from the disease head, empty if it did not run."""
    output = report.head_outputs.get("disease")
    if not isinstance(output, dict):
        return [], []
    labels = [str(label) for label in output]
    return labels, [float(output[label]) for label in labels]


def drought_markers(report: AnalysisReport) -> tuple[list[str], list[float]]:
    """Drought-stage marker names and scores, empty if the drought-stage head did not run."""
    output = report.head_outputs.get("drought_stage")
    if not isinstance(output, dict):
        return [], []
    markers = output.get("markers")
    if not isinstance(markers, dict):
        return [], []
    names = [str(name) for name in markers]
    return names, [float(markers[name]) for name in names]


def observation_table(observations: Sequence[Observation]) -> list[dict[str, object]]:
    """Time-ordered rows for a plant's observation series."""
    return [
        {"timestamp": obs.timestamp, "stress_score": round(obs.stress_score, 4)}
        for obs in observations
    ]


def quality_banner(report: AnalysisReport) -> str | None:
    """A one-line reliability warning for the analyze tab, or None when the input looks fine."""
    if report.quality.usable:
        return None
    return "Low input quality: " + "; ".join(report.quality.warnings)


def timing_rows(report: AnalysisReport) -> list[dict[str, object]]:
    """Per-stage wall-clock timing rows in pipeline order."""
    return [{"stage": stage, "ms": round(ms, 1)} for stage, ms in report.timing_ms.items()]


def forecast_points(forecast: Forecast) -> tuple[list[int], list[float]]:
    """Horizon steps and projected scores for the forecast line, in ascending horizon order."""
    horizons = sorted(forecast.projected_scores)
    return horizons, [forecast.projected_scores[horizon] for horizon in horizons]


def forecast_band(forecast: Forecast) -> tuple[list[int], list[float], list[float]]:
    """Horizon steps and their lower and upper interval bounds, for the projection's shaded band.

    Only horizons that carry an interval are returned, so a degenerate forecast draws no band.
    """
    horizons = [h for h in sorted(forecast.projected_scores) if h in forecast.lower]
    lower = [forecast.lower[h] for h in horizons]
    upper = [forecast.upper[h] for h in horizons]
    return horizons, lower, upper


def survival_curve_points(
    fit: SurvivalFit,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """The cohort survival curve as four aligned lists: times, survival, lower band, upper band.

    The band lists are empty when the curve carries no confidence interval.
    """
    curve = fit.curve
    return (
        list(curve.times),
        list(curve.survival),
        list(curve.lower),
        list(curve.upper),
    )


def plant_survival_metrics(fit: SurvivalFit, plant_id: str) -> dict[str, object]:
    """One plant's median time-to-wilt, its band, and the basis, for a dashboard metric."""
    plant = fit.per_plant.get(plant_id)
    if plant is None:
        return {"median": None, "lower": None, "upper": None, "basis": "unavailable"}
    return {
        "median": plant.median,
        "lower": plant.lower,
        "upper": plant.upper,
        "basis": plant.basis,
    }

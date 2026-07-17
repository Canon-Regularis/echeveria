"""Streamlit dashboard: a basic, sleek water-stress terminal.

Needs the 'dashboard' extra: pip install -e ".[dashboard]". Launch it with
``phytovision dashboard``, which runs ``streamlit run`` on this module with a neutral-dark theme.
The analysed pipeline mirrors the API: ``Pipeline.default()`` unless ``PHYTOVISION_CONFIG`` or
``PHYTOVISION_MODEL_PATH`` is set.

The UI has two tabs. ANALYZE reads one image and shows the verdict, overlay, drivers, the
placeholder disease panel, timing, and features. TEMPORAL charts stress over time for a plant from a
CSV manifest.

Streamlit and plotly are imported lazily inside the render functions, so the pure helpers below stay
importable, and therefore testable, with only the base dependencies installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from phytovision.exceptions import PhytoVisionError
from phytovision.io import decode_rgb_bytes
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.models.survival import SurvivalFit
from phytovision.pipeline import Pipeline
from phytovision.serving import attach_heads, engine_from_env
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


_TERMINAL_CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1500px; }
h1, h2, h3 { text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricLabel"] { text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.75; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
</style>
"""

_DARK_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 8, "r": 8, "t": 8, "b": 8},
}


def render() -> None:  # pragma: no cover: exercised only inside a running Streamlit server
    """The Streamlit entry point. Streamlit executes this module top to bottom on each rerun."""
    import streamlit as st

    st.set_page_config(page_title="phytovision", page_icon="🌱", layout="wide")
    st.markdown(_TERMINAL_CSS, unsafe_allow_html=True)
    st.markdown("### PHYTOVISION // EXPLAINABLE WATER-STRESS TERMINAL")

    engine, conformal = engine_from_env()
    analyze_tab, temporal_tab = st.tabs(["ANALYZE", "TEMPORAL"])
    with analyze_tab:
        _render_analyze_tab(engine, conformal)
    with temporal_tab:
        _render_temporal_tab(engine)


def _render_analyze_tab(
    engine: Pipeline, conformal: SplitConformalClassifier | None
) -> None:  # pragma: no cover: Streamlit UI
    import plotly.graph_objects as go
    import streamlit as st

    from phytovision.visualize import render_overlay, render_saliency_overlay

    st.caption("Upload a plant image. The verdict, overlay, and drivers behind it appear below.")
    upload = st.file_uploader(
        "Plant image", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="analyze_upload"
    )
    if upload is None:
        st.info("Waiting for an image.")
        return

    # The terminal always shows the disease and drought-stage panels.
    engine = attach_heads(engine, disease=True, drought_stage=True)
    try:
        image = decode_image(upload.getvalue())
        report = engine.analyze(image)
    except PhytoVisionError as exc:
        st.error(str(exc))
        return

    stress = report.stress
    st.subheader(f"Verdict: {stress.label.upper()}")
    score, confidence, regions, model = st.columns(4)
    score.metric("stress score", f"{stress.score:.2f}")
    confidence.metric("confidence", f"{stress.confidence:.2f}")
    regions.metric(f"regions ({report.regions.kind})", len(report.regions))
    model.metric("model", stress.model_name)

    banner = quality_banner(report)
    if banner:
        st.warning(banner)

    if conformal is not None:
        label_set = conformal.predict_set(report.plant_features)
        coverage = round((1.0 - label_set.alpha) * 100)
        members = ", ".join(label_set.labels) or "(empty)"
        st.caption(f"Conformal set ({coverage}% coverage): {members}")

    stage = report.head_outputs.get("drought_stage")
    if isinstance(stage, dict):
        with st.container(border=True):
            st.markdown("**DROUGHT STAGE**: literature-motivated rule set, not a diagnosis")
            st.metric("stage", str(stage.get("stage", "?")).upper())
            st.caption(f"basis: {stage.get('basis', '')}")
            names, scores = drought_markers(report)
            if names:
                figure = go.Figure(go.Bar(x=scores, y=names, orientation="h"))
                figure.update_layout(
                    xaxis={"range": [0, 1]}, height=32 * len(names) + 120, **_DARK_LAYOUT
                )
                st.plotly_chart(figure, use_container_width=True)

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**INPUT / OVERLAY**")
        st.image(image, caption="input", use_container_width=True)
        st.image(render_overlay(image, report), caption="overlay", use_container_width=True)
        st.image(
            render_saliency_overlay(image, report, engine.model),
            caption="pigment saliency (RGB proxy)",
            use_container_width=True,
        )
    with right.container(border=True):
        st.markdown("**DRIVERS**")
        features, contributions = contribution_series(report)
        if features:
            figure = go.Figure(go.Bar(x=contributions, y=features, orientation="h"))
            figure.update_layout(
                yaxis={"autorange": "reversed"}, height=32 * len(features) + 120, **_DARK_LAYOUT
            )
            st.plotly_chart(figure, use_container_width=True)
        st.dataframe(reason_rows(report), use_container_width=True, hide_index=True)

    disease_col, timing_col = st.columns(2)
    with disease_col.container(border=True):
        st.markdown("**DISEASE**: placeholder, not a validated diagnostic")
        labels, probabilities = disease_series(report)
        for label, probability in zip(labels, probabilities, strict=True):
            st.metric(label, f"{probability:.2f}")
    with timing_col.container(border=True):
        st.markdown("**TIMING (ms)**")
        rows = timing_rows(report)
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No per-stage timing recorded.")

    with st.container(border=True):
        st.markdown("**FEATURES**")
        table = [
            {"feature": key, "value": round(value, 4)}
            for key, value in sorted(report.plant_features.defined().items())
        ]
        st.dataframe(table, use_container_width=True, hide_index=True)


def _render_temporal_tab(engine: Pipeline) -> None:  # pragma: no cover: Streamlit UI
    import plotly.graph_objects as go
    import streamlit as st

    from phytovision.datasets.manifest import CsvManifestLoader
    from phytovision.models.survival import fit_cohort_survival
    from phytovision.registries import FORECASTERS, SURVIVAL_MODELS
    from phytovision.temporal import (
        DEFAULT_HORIZONS,
        build_history,
        pigment_early_warning,
        stress_trend,
    )

    st.caption(
        "Point at a CSV manifest with plant_id and timestamp columns to chart stress trends."
    )
    manifest_path = st.text_input("Manifest path (.csv or .tsv)", key="manifest_path")
    images_root = st.text_input(
        "Images root (optional, defaults to the manifest folder)", key="images_root"
    )
    if not manifest_path:
        st.info("Enter a manifest path to load a time series.")
        return

    try:
        loader = CsvManifestLoader(manifest_path, images_root or None)
        history = build_history(engine, loader)
    except (OSError, PhytoVisionError) as exc:
        st.error(str(exc))
        return
    if not history.plant_ids:
        st.warning("No samples with both a plant_id and a timestamp were found.")
        return

    plant_id = st.selectbox("Plant", history.plant_ids)
    series = history.series_for(plant_id)
    trend = stress_trend(plant_id, series)

    direction, slope, count = st.columns(3)
    direction.metric("trend", trend.direction.upper())
    slope.metric("slope / step", f"{trend.slope:+.3f}")
    count.metric("observations", trend.n)

    warning = pigment_early_warning(plant_id, series)
    if warning.flagged:
        st.warning(
            f"EARLY WARNING (RGB pigment proxy, not a measurement): {warning.note} "
            f"(pigment slope {warning.pigment_slope:+.3f})."
        )

    figure = go.Figure(
        go.Scatter(
            x=[obs.timestamp for obs in series],
            y=[obs.stress_score for obs in series],
            mode="lines+markers",
        )
    )
    figure.update_layout(yaxis={"title": "stress score", "range": [0, 1]}, **_DARK_LAYOUT)
    st.plotly_chart(figure, use_container_width=True)

    method = st.selectbox("Forecaster", FORECASTERS.names(), key="forecaster")
    scores = [obs.stress_score for obs in series]
    try:
        forecast = FORECASTERS.create(method).forecast(scores, DEFAULT_HORIZONS, plant_id)
    except ImportError as exc:
        st.warning(f"{exc} Falling back to the linear-trend forecaster.")
        forecast = FORECASTERS.create("linear-trend").forecast(scores, DEFAULT_HORIZONS, plant_id)
    to_stressed = forecast.steps_to_stressed
    steps_col, confidence_col = st.columns(2)
    steps_col.metric("steps to stressed", to_stressed if to_stressed is not None else "n/a")
    confidence_col.metric("forecast confidence", f"{forecast.confidence:.2f}")
    band_pct = f"{forecast.interval_level:.0%}"
    st.caption(
        f"Forecast ({forecast.method}) is an RGB-proxy extrapolation, not a validated prediction; "
        f"the shaded band is a {band_pct} prediction interval, not a measurement."
    )
    steps, projected = forecast_points(forecast)
    if steps:
        # Anchor at the fitted trend level, not the raw last reading, so the line stays on-trend.
        projection = go.Figure()
        band_steps, lower, upper = forecast_band(forecast)
        if band_steps:
            projection.add_trace(go.Scatter(x=band_steps, y=upper, mode="lines", line={"width": 0}))
            projection.add_trace(
                go.Scatter(x=band_steps, y=lower, mode="lines", line={"width": 0}, fill="tonexty")
            )
        projection.add_trace(
            go.Scatter(
                x=[0, *steps],
                y=[forecast.current_level, *projected],
                mode="lines+markers",
                line={"dash": "dash"},
            )
        )
        projection.update_layout(
            yaxis={"title": "projected stress", "range": [0, 1]},
            xaxis={"title": "steps ahead"},
            showlegend=False,
            **_DARK_LAYOUT,
        )
        st.plotly_chart(projection, use_container_width=True)

    _render_survival(st, go, history, plant_id, SURVIVAL_MODELS, fit_cohort_survival)

    st.dataframe(observation_table(series), use_container_width=True, hide_index=True)


def _render_survival(
    st: Any, go: Any, history: Any, plant_id: str, registry: Any, fit_cohort: Any
) -> None:  # pragma: no cover: Streamlit UI
    st.markdown("#### SURVIVAL")
    model = st.selectbox("Survival model", registry.names(), key="survival_model")
    try:
        fit = fit_cohort(history, model)
    except ImportError:
        st.warning('Survival needs the stats extra: pip install -e ".[stats]"')
        return

    metrics = plant_survival_metrics(fit, plant_id)
    median = metrics["median"]
    band = f"{metrics['lower']} to {metrics['upper']} ({metrics['basis']})"
    # A cohort-km row broadcasts the Kaplan-Meier 95% median CI; only the covariate models report an
    # interquartile time band. Labelling both "central 50%" would understate the KM interval.
    band_kind = (
        "cohort 95% CI (Kaplan-Meier)" if metrics["basis"] == "cohort-km" else "central 50% band"
    )
    st.metric(
        "median time to wilt (obs steps)",
        "n/a" if median is None else f"{median:.1f}",
        help=f"{band_kind}: {band}",
    )

    times, survival, lower, upper = survival_curve_points(fit)
    figure = go.Figure()
    if lower and upper:
        figure.add_trace(go.Scatter(x=times, y=upper, mode="lines", line={"width": 0}))
        figure.add_trace(
            go.Scatter(x=times, y=lower, mode="lines", line={"width": 0}, fill="tonexty")
        )
    figure.add_trace(go.Scatter(x=times, y=survival, mode="lines", line={"shape": "hv"}))
    if median is not None:
        figure.add_vline(x=median, line={"dash": "dash"})
    figure.update_layout(
        yaxis={"title": "cohort survival", "range": [0, 1]},
        xaxis={"title": "observation steps"},
        showlegend=False,
        **_DARK_LAYOUT,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Median time-to-wilt is a synthetic-trained, RGB-proxy distribution over time-to-event; "
        "the shaded band is the Kaplan-Meier confidence interval, not a measurement; the reported "
        "concordance is in-sample and optimistic; it is indicative, not a validated prognosis."
    )


if __name__ == "__main__":  # pragma: no cover: the streamlit runner sets __name__ to __main__
    render()

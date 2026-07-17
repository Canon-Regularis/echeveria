"""The TEMPORAL tab: stress over time for one plant, with a forecast band and a survival curve.

Streamlit, plotly, and the temporal machinery are imported lazily inside the render functions, so
importing this module needs only the base dependencies.
"""

from __future__ import annotations

from typing import Any

from phytovision.dashboard.helpers import (
    forecast_band,
    forecast_points,
    observation_table,
    plant_survival_metrics,
    survival_curve_points,
)
from phytovision.dashboard.theme import DARK_LAYOUT
from phytovision.exceptions import PhytoVisionError
from phytovision.pipeline import Pipeline


def render_temporal_tab(engine: Pipeline) -> None:  # pragma: no cover: Streamlit UI
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
    figure.update_layout(yaxis={"title": "stress score", "range": [0, 1]}, **DARK_LAYOUT)
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
            **DARK_LAYOUT,
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
        **DARK_LAYOUT,
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Median time-to-wilt is a synthetic-trained, RGB-proxy distribution over time-to-event; "
        "the shaded band is the Kaplan-Meier confidence interval, not a measurement; the reported "
        "concordance is in-sample and optimistic; it is indicative, not a validated prognosis."
    )

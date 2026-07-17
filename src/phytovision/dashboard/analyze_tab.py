"""The ANALYZE tab: one image to a verdict, overlay, drivers, the placeholder heads, and features.

Streamlit and plotly are imported lazily inside the render function, so importing this module needs
only the base dependencies.
"""

from __future__ import annotations

from phytovision.dashboard.helpers import (
    contribution_series,
    decode_image,
    disease_series,
    drought_markers,
    quality_banner,
    reason_rows,
    timing_rows,
)
from phytovision.dashboard.theme import DARK_LAYOUT
from phytovision.exceptions import PhytoVisionError
from phytovision.models.conformal import SplitConformalClassifier
from phytovision.pipeline import Pipeline
from phytovision.serving import attach_heads


def render_analyze_tab(
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

    # The terminal always shows the disease, drought-stage, and physiology panels.
    engine = attach_heads(engine, disease=True, drought_stage=True, physiology=True)
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
                    xaxis={"range": [0, 1]}, height=32 * len(names) + 120, **DARK_LAYOUT
                )
                st.plotly_chart(figure, use_container_width=True)

    physiology = report.head_outputs.get("physiology")
    if isinstance(physiology, dict):
        with st.container(border=True):
            st.markdown("**PHYSIOLOGY**: crude RGB proxies, not measured physiology")
            potential, conductance, transpiration = st.columns(3)
            potential.metric(
                "water potential", f"{physiology.get('water_potential_proxy', 0.0):.2f}"
            )
            conductance.metric(
                "stomatal conductance", f"{physiology.get('stomatal_conductance_proxy', 0.0):.2f}"
            )
            transpiration.metric(
                "transpiration", f"{physiology.get('transpiration_proxy', 0.0):.2f}"
            )
            st.caption(f"basis: {physiology.get('basis', '')}")

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
                yaxis={"autorange": "reversed"}, height=32 * len(features) + 120, **DARK_LAYOUT
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
